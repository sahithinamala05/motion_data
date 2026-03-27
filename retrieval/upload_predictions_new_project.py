"""
Upload pre-label annotations to a NEW Label Studio project.

Fetches tasks from the new project, maps video filenames to new task IDs,
then uploads annotations.

Usage:
    python upload_predictions_new_project.py \
        --url https://app.humansignal.com \
        --token YOUR_REFRESH_TOKEN \
        --project-id 245163
"""
import argparse
import base64
import json
import os
import re
import requests

SCORE_THRESHOLD = 0.9

INPUT_FILES = [
    ("vis_out/inference/detections_filtered_videos_feat.json", "split"),
    ("vis_out/inference_clean_hand/detections_filtered_videos_feat.json", None),
]


def get_access_token(url, refresh_token):
    resp = requests.post(
        f"{url.rstrip('/')}/api/token/refresh",
        json={"refresh": refresh_token},
    )
    resp.raise_for_status()
    return resp.json()["access"]


def fetch_all_tasks(url, headers, project_id, refresh_token):
    """Fetch all tasks from a project, handling pagination."""
    tasks = []
    page = 1
    while True:
        resp = requests.get(
            f"{url.rstrip('/')}/api/tasks",
            headers=headers,
            params={"project": project_id, "page": page, "page_size": 1000},
        )
        if resp.status_code == 401:
            access_token = get_access_token(url, refresh_token)
            headers["Authorization"] = f"Bearer {access_token}"
            resp = requests.get(
                f"{url.rstrip('/')}/api/tasks",
                headers=headers,
                params={"project": project_id, "page": page, "page_size": 1000},
            )
        resp.raise_for_status()
        data = resp.json()
        # Response could be a dict with "tasks" key or a list
        if isinstance(data, list):
            batch = data
        else:
            batch = data.get("tasks", [])
            # Debug: print keys on first page to understand structure
            if page == 1:
                print(f"  Response keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")
                print(f"  Total reported: {data.get('total', 'N/A')}")
        tasks.extend(batch)
        total = data.get("total", 0) if isinstance(data, dict) else len(tasks)
        print(f"  Page {page}: got {len(batch)} tasks (total so far: {len(tasks)}/{total})")
        if len(tasks) >= total or len(batch) == 0:
            break
        page += 1
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--project-id", required=True, type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Exchanging refresh token for access token...")
    access_token = get_access_token(args.url, args.token)
    print("Got access token.")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    # Step 1: Fetch all tasks from the new project
    print(f"Fetching tasks from project {args.project_id}...")
    tasks = fetch_all_tasks(args.url, headers, args.project_id, args.token)
    print(f"  Found {len(tasks)} tasks.")

    # Step 2: Build video_name -> new_task_id mapping
    video_to_task = {}
    for task in tasks:
        video_url = task.get("data", {}).get("video", "")
        # Extract base64-encoded fileuri and decode to get S3 path
        match = re.search(r'fileuri=([A-Za-z0-9+/=]+)', video_url)
        if match:
            s3_path = base64.b64decode(match.group(1)).decode("utf-8")
            video_basename = os.path.basename(s3_path)
        else:
            video_basename = os.path.basename(video_url)
        video_name = os.path.splitext(video_basename)[0]
        video_to_task[video_name] = task["id"]

    print(f"  Mapped {len(video_to_task)} video names to task IDs.")
    # Debug: show sample mappings
    sample_keys = sorted(video_to_task.keys())[:3]
    for k in sample_keys:
        print(f"    e.g. '{k}' -> {video_to_task[k]}")
    # Show raw video URL from first task
    if tasks:
        print(f"    Raw video URL example: {tasks[0].get('data', {}).get('video', 'N/A')}")

    # Step 3: Collect predictions with new task IDs
    predictions = []
    unmapped = set()
    for f, default_action in INPUT_FILES:
        data = json.load(open(f))
        for video_key, video_data in data.items():
            action = video_data.get("action", default_action)
            for det in video_data["detections"]:
                if det["score"] > SCORE_THRESHOLD:
                    video_name = video_data["video_name"]
                    new_task_id = video_to_task.get(video_name)
                    if new_task_id is None:
                        unmapped.add(video_name)
                        continue
                    predictions.append({
                        "task_id": new_task_id,
                        "result": [
                            {
                                "from_name": "videoLabels",
                                "to_name": "video",
                                "type": "timelinelabels",
                                "value": {
                                    "ranges": [
                                        {
                                            "start": det["query_start"],
                                            "end": det["query_end"],
                                        }
                                    ],
                                    "timelinelabels": [action],
                                },
                            }
                        ],
                        "score": det["score"],
                    })

    if unmapped:
        print(f"  WARNING: {len(unmapped)} videos not found in new project")
        for v in sorted(unmapped)[:5]:
            print(f"    {v}")

    print(f"  Collected {len(predictions)} annotations to upload.")

    if args.dry_run:
        if not predictions:
            print("\nDry run: no predictions to upload!")
            return
        pred = predictions[0]
        print(f"\nDry run: task {pred['task_id']}")
        print(json.dumps(pred, indent=2))
        return

    # Step 4: Upload annotations
    print(f"Uploading {len(predictions)} annotations...")
    success = 0
    failed = 0
    for i, pred in enumerate(predictions):
        task_id = pred["task_id"]
        payload = {
            "result": pred["result"],
            "was_cancelled": False,
            "ground_truth": False,
        }
        resp = requests.post(
            f"{args.url.rstrip('/')}/api/tasks/{task_id}/annotations",
            headers=headers,
            json=payload,
        )
        if resp.status_code == 201:
            success += 1
        elif resp.status_code == 401:
            print(f"  Token expired at item {i+1}, refreshing...")
            access_token = get_access_token(args.url, args.token)
            headers["Authorization"] = f"Bearer {access_token}"
            resp = requests.post(
                f"{args.url.rstrip('/')}/api/tasks/{task_id}/annotations",
                headers=headers,
                json=payload,
            )
            if resp.status_code == 201:
                success += 1
            else:
                failed += 1
                print(f"  FAIL task {task_id}: {resp.status_code} {resp.text[:500]}")
        else:
            failed += 1
            print(f"  FAIL task {task_id}: {resp.status_code} {resp.text[:500]}")

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(predictions)} (ok={success}, fail={failed})")

    print(f"\nDone: {success} succeeded, {failed} failed out of {len(predictions)}")


if __name__ == "__main__":
    main()
