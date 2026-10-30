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

## Open items (updated 2026-10-30: the A100 VM was terminated; a new RTX PRO 6000 VM replaces it)

| # | Item | What to do | Then | Status |
|---|---|---|---|---|
| 0 | **New VM** | Create the RTX PRO 6000 VM with **≥ 1.5 TB disk** (2 TB is safer), ideally ≥ 32 vCPUs. | Send Claude the IP and SSH user. Claude runs `bash src/vm/start_vm.sh <ip>`; everything else starts by itself. | OPEN |
| 1 | **Waymo login (again)** | The old login lived on the deleted VM. About 10 minutes after start, in your own terminal: `ssh gpunet@<ip>` then `~/rs3d/bin/waymo_login.sh`. Sign in with the same Google account and paste the code **into that terminal, not the chat**. | The download starts by itself. To drop Waymo instead: `touch ~/rs3d/skip_waymo`. | OPEN |
| 2 | **Aria Digital Twin** | Done: `ADT_download_urls.json` is in the project folder (links valid until 2026-11-28; kept out of git). | Nothing. | DONE |
| 3 | **ScanNet++** (iPhone RS video + laser scans) | Open https://scannetpp.mlsg.cit.tum.de/scannetpp/, create an account, and submit the application. Approval usually takes days. | Save the download token as `scannetpp_token.txt` in the project folder, or paste it to Claude. | OPEN |
| 4 | **VGGT-Ω weights** (gated on Hugging Face) | Accept the gate at https://huggingface.co/facebook/VGGT-Omega (FAIR non-commercial licence), then create a *read* token at https://huggingface.co/settings/tokens. | Put it in `~/rs3d/.hf/token` on the VM, or paste it to Claude. VGGT-Ω joins the runs automatically when the file appears. To drop it: `touch ~/rs3d/skip_vggt_omega`. | OPEN |
| 5 | **Before switching the VM off** | Tell Claude first: it copies every result file to the laptop (`src/vm/pull_results.sh`, also run regularly while the VM is up). | — | RULE |

## Not needed (for the record)

- **HM3D, Matterport3D and 3D-FRONT** would also need sign-ups. Replica covers the mesh need, so these are not requested.
- **ARKitScenes**: its terms are accepted just by using it. It is not requested, because ScanNet++ covers the same need better.
