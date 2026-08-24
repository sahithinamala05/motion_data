"""Quick single-segment SAM3 box-prompt trajectory test for the split case."""
import os, json, numpy as np, torch, cv2
from sam3.model_builder import build_sam3_video_predictor

VIDEO = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/good_quality_rounds/2025-10-01_06-05-15_009910_011977.mp4"
# seg (837,876), keyframe 837, two card boxes xyxy in 1280x720
W, H = 1280, 720
kf_frame = 837
card_boxes_xyxy = [[767,516,835,554],[794,530,870,574]]
start_f, end_f = 837, 876

def xyxy_to_norm_xywh(b):
    x1,y1,x2,y2=b
    return [x1/W, y1/H, (x2-x1)/W, (y2-y1)/H]

def center_norm(b):
    x1,y1,x2,y2=b
    return [ (x1+x2)/2/W, (y1+y2)/2/H ]

centers = [center_norm(b) for b in card_boxes_xyxy]
print("norm centers:", centers, flush=True)

predictor = build_sam3_video_predictor()
sid = predictor.handle_request(dict(type="start_session", resource_path=VIDEO))["session_id"]
for oid, c in enumerate(centers):
    resp = predictor.handle_request(dict(
        type="add_prompt", session_id=sid, frame_index=kf_frame,
        obj_id=oid, points=[c], point_labels=[1], rel_coordinates=True))
    out = resp["outputs"]
    print(f"after add obj {oid}: ids={out['out_obj_ids']}", flush=True)

n_track = end_f - start_f + 6
traj = {}
for r in predictor.handle_stream_request(dict(
        type="propagate_in_video", session_id=sid,
        start_frame_index=kf_frame, max_frame_num_to_track=n_track,
        propagation_direction="both")):
    fi = r["frame_index"]
    if not (start_f <= fi <= end_f):
        continue
    o = r["outputs"]
    ids = o["out_obj_ids"].tolist() if hasattr(o["out_obj_ids"],"tolist") else list(o["out_obj_ids"])
    boxes_o = o["out_boxes_xywh"]
    for j,oid in enumerate(ids):
        bx = boxes_o[j]; bx = bx.tolist() if hasattr(bx,"tolist") else list(bx)
        x,y,w,h=[float(v) for v in bx]
        if max(x,y,w,h)<=1.5:
            x,w=x*W,w*W; y,h=y*H,h*H
        cx,cy=x+w/2,y+h/2
        traj.setdefault(int(oid),[]).append((fi,round(cx,1),round(cy,1)))
predictor.handle_request(dict(type="close_session", session_id=sid))

for oid,pts in traj.items():
    pts.sort()
    print(f"obj {oid}: {len(pts)} frames, first {pts[0]} last {pts[-1]}", flush=True)
print("DONE")
