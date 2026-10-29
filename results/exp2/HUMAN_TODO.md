# EXP2: things only a human can do

These are steps that need your account, your signature or your acceptance of a licence (EXP2 rule 8). Everything else keeps running while they are open.

## Decisions already made (2026-10-29)

- **Replica**
  - Approved by you to use under the *Replica Dataset Research Terms* (non-commercial research).
  - Being downloaded to `~/rs3d/data/replica`.
- **Creative Commons datasets**
  - Approved by you, with each licence recorded in DATA_AUDIT.md:
    - ETH3D (CC BY-NC-SA 4.0)
    - ADVIO (CC BY-NC 4.0)
    - TartanAir V2 (CC BY 4.0)
- **Sign-ups you said you will do:** Waymo, Aria Digital Twin and ScanNet++. The steps are below.

## Open items

| # | Item | What to do | Then give me | Status |
|---|---|---|---|---|
| 1 | **Waymo Open Dataset** (real RS, per-image readout timing, LiDAR) | Open https://waymo.com/open, click Download, sign in with Google and accept the *Waymo Dataset License Agreement for Non-Commercial Use*. | In your own terminal run `ssh gpunet@198.145.127.129` then `~/rs3d/bin/waymo_login.sh`. Open the link it prints, sign in with the same Google account, and paste the code back. The credential stays in `~/rs3d/.gcloud`. Then tell Claude "Waymo ready". | OPEN |
| 2 | **Aria Digital Twin** (real RS, 5 ms readout, mocap, dense depth) | Open https://www.projectaria.com/datasets/adt/, fill in the access form (email) and accept the ADT licence. You get a file `ADT_download_urls.json`, by email or as a page download. | Save that file into the project folder (`Downloads/3D_CVPR_Exp_2_Rolling_Shutter/`) and tell Claude "ADT ready". Claude copies it to the VM and downloads with the official `aria_dataset_downloader` (already installed). | OPEN |
| 3 | **ScanNet++** (iPhone RS video + laser scans) | Open https://scannetpp.mlsg.cit.tum.de/scannetpp/, create an account, and submit the application with your institution and agreement to the Terms of Use. Approval usually takes days. | When the approval email arrives, log in and copy your download token. Save it as `scannetpp_token.txt` in the project folder, or paste it to Claude. | OPEN |
| 4 | **VGGT-Ω weights** (newest strong model, gated on Hugging Face) | Accept the gate at https://huggingface.co/facebook/VGGT-Omega (FAIR non-commercial licence). Then create a *read* token at https://huggingface.co/settings/tokens. | Put the token in `~/rs3d/.hf/token` on the VM, or paste it to me. | OPEN |
| 5 | **GPU sharing** | The other project's experiment (tmux session `e2eth3d`, `~/3D-CVPR_Exp`, `~/data`, `~/miniconda3`) is on the same VM. I don't touch it and I check free VRAM before every launch. | Nothing, unless you want a fixed VRAM split. | INFO |
| 6 | **Local C: drive is full** | Your laptop's C: drive has under 300 MB free. Git and the report build may fail. | Free a few GB when you can. All heavy data stays on the VM. | OPEN |

## Not needed (for the record)

- **HM3D, Matterport3D and 3D-FRONT** would also need sign-ups. Replica covers the mesh need, so these are not requested.
- **ARKitScenes**: its terms are accepted just by using it. It is not requested, because ScanNet++ covers the same need better.
