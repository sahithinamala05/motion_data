# sam_tracking

SAM 3 tracking of playing cards + table objects, and SRSS trajectory post-processing.

- `tracking/` — SAM3 tracking runs: batch card/combined tracking (`batch_cards_1000.py`,
  `batch_clips10.py`, `batch_combined_1000.py`), single-video tracking (`track_combined.py`,
  `track_table_objects.py`, `sam_track_srss.py`), and split-card trajectory extraction
  (`split_test.py`, `split_track_clustered.py`, `split_trajectory_sam.py`).
- `trajectory_postproc/` — post-processing of SRSS SAM trajectories: endpoints
  (`add_srss_endpoints.py`), reorder (`reorder_srss_trajectories.py`), re-verify
  (`reverify_split_traj.py`), annotation update (`update_srss_annotations_from_tracking.py`),
  viz (`viz_srss_trajectory.py`), and segment extraction (`extract_red_card_segments.py`).
