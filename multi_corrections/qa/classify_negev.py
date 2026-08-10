"""Split negative-evidence clips into LOBBY (menu grid = many small bodies) vs real TABLE (2-4 bodies)."""
import os, pickle, numpy as np
ORIG="/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor_rem"
stems=[l.strip() for l in open("/tmp/claude-1000/-home-ubuntu/88cc26b9-b104-4dd1-b365-b88dd225bc5c/scratchpad/negev.txt") if l.strip()]
lobby=[]; table=[]
for s in stems:
    p=os.path.join(ORIG,f"{s}_dwpose.pkl")
    if not os.path.exists(p): continue
    try: d=pickle.load(open(p,'rb'))
    except Exception: continue
    maxb=0
    for fr in d:
        b=fr['pose']['bodies']
        if b is None: continue
        maxb=max(maxb, np.asarray(b['subset']).shape[0])
    if maxb>=6: lobby.append(s)
    elif 2<=maxb<=4: table.append(s)
    if len(lobby)>=10 and len(table)>=10: break
sp="/tmp/claude-1000/-home-ubuntu/88cc26b9-b104-4dd1-b365-b88dd225bc5c/scratchpad"
open(f"{sp}/case_lobby.txt","w").write("\n".join(lobby[:10]))
open(f"{sp}/case_table.txt","w").write("\n".join(table[:10]))
print(f"lobby(>=6 bodies): {len(lobby[:10])}  table(2-4 bodies): {len(table[:10])}")
print("lobby:", lobby[:10])
print("table:", table[:10])
