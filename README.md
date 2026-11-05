# Media Duplicate Finder
A Windows desktop program that finds duplicate and look-alike photos and videos, so you can review them and move the extra copies into a separate folder.

## What it does
- Finds exact duplicate files by comparing their SHA-256 fingerprints.
- Finds resized or re-compressed copies of the same photo, and possible cropped or edited versions.
- Finds the same video saved at different sizes, such as 1080p and 4K.
- Shows each group of matches with one file marked **KEEP**. That file is never moved.
- Moves only the files you tick into a review folder. It never deletes your files.
- Writes a CSV list (a "manifest") of every move, so you can undo the move later.

## What you need
- Windows 10 or 11 (it also runs on macOS and Linux).
- Python 3.10 or newer.
- Optional: an NVIDIA graphics card with a current driver, if you want the AI matching to run fast.
- Optional: PyTorch for AI matching. This is a large download (about 2.5 GB for the NVIDIA version).

## Install
1. Install Python from python.org. During setup, tick "Add Python to PATH".
2. Download this project and unzip it.
3. Open a command prompt in the project folder.
4. Run `pip install -r requirements.txt`
5. Optional, for AI matching: start the program and click **Install CUDA PyTorch** (NVIDIA cards) or **Install CPU PyTorch** (other computers). The program asks before it starts the download. Close and reopen the program when it finishes.

## How to use
1. Start the program: `py media_duplicate_finder.py`
2. On the **Setup / Scan** tab, click **Add cleanup folder** and pick a folder or drive to scan. You can also use **Add all drives** or **Add common photo/video folders**.
3. Optional: click **Add protected folder** to add a "master" folder. Files in protected folders are kept and never moved.
4. Choose your options under **2) Scan options**. The defaults are careful and are a good start.
5. Click **Start scan**. The progress bar and the **Log** tab show what is happening.
6. When the scan ends, open the **Results** tab. Exact copies are on the **Duplicates** tab. Look-alike files are on the **Similar photos / videos** tab.
7. Tick the files you want to move. You can use **Check exact duplicate moves** or **Check lower-quality similar files** to tick them for you.
8. Click **Move checked files to review folder** (or **Move checked exact duplicates only**) and choose a review folder. The review folder must be outside the folders you scanned.
9. Look through the review folder. Delete the files yourself when you are sure you do not need them.

## Settings and options
- **Scan all subfolders recursively**: also scan folders inside the chosen folders.
- **Skip Windows/system/program folders**: skips folders such as Windows, Program Files and AppData.
- **Find exact duplicates with SHA-256**: finds files that are exactly the same.
- **Include all file types for exact duplicates**: also checks documents and other files, not only photos and videos.
- **Find conservative resized/compressed photo copies** and **Find conservative similar videos / different resolution copies**: find look-alike files.
- **Find possible cropped/edited photo suggestions (review-only, unchecked)**: shows possible crops. They are never ticked for you.
- **Pre-check similar photo/video matches for moving (not recommended)**: ticks look-alike files automatically.
- **Use AI/deep embeddings as extra confirmation only**: uses a ResNet50 image model to double-check matches. Needs PyTorch.
- **Acceleration mode**: Auto, NVIDIA (require CUDA), or Laptop / CPU-only.
- Threshold boxes: higher hash numbers and lower embedding numbers find more matches, but with more mistakes.
- **Max files for test scan**: stop after this many files (0 means no limit).
- **Disk guard**: stop if the free space on a drive gets too low.
- **Preserve folder hierarchy**: keeps the original folder layout inside the review folder.
- **Remove empty source folders**: removes a source folder only if it becomes completely empty after the move.
- **File** menu: **Save scan results cache...** and **Load scan results cache...** let you save results and open them later without scanning again.

## Safety notes
- The program never deletes your photos or videos. It only moves the files you tick.
- The **KEEP** file in each group is never moved, even if it is ticked.
- Each move is checked. On the same drive, the file size is checked after the move. When moving to another drive, the program copies the file, compares the size and SHA-256 fingerprint, and removes the original only if they match. If a check fails, the original stays where it was and the manifest row says `verify_failed_kept_source`.
- It never overwrites a file. If a name is taken, it adds `__1`, `__2` and so on.
- To undo a move, go to the **Tools** tab and click **Restore moved files from manifest...**, then pick the `move_manifest_....csv` file in your review folder.
- **Remove empty folders now** on the **Tools** tab removes empty folders only. It asks first and saves a CSV list in your user folder. Use **Preview empty folders** to see the list first.
- The install buttons run `pip install` only after you say yes.

## Troubleshooting
- **"Some required Python packages are missing"**: run `pip install -r requirements.txt` in the project folder, then start the program again.
- **The AI option does nothing**: PyTorch is not installed. Click **Check NVIDIA / PyTorch** to see the status.
- **NVIDIA card not used**: click **Install CUDA PyTorch**, then close and reopen the program.
- **The first AI scan wants to download a model**: click **Preload ResNet50** before scanning (about 100 MB).
- **The program says the review folder is inside a scanned folder**: choose a review folder somewhere else, so moved files are not found again by the next scan.

## How it works
Exact duplicates are found by grouping files by size and then by SHA-256 hash, using several threads. Look-alike photos are found with perceptual hashes (pHash, aHash, dHash, wHash from the ImageHash library) stored in BK-trees, and each candidate pair must pass extra checks before it is grouped. Videos are compared by sampling frames with OpenCV and comparing their hashes. Optional ResNet50 embeddings from PyTorch/torchvision add a second opinion, on the GPU when CUDA is available. The interface is built with PySide6 (Qt), and long jobs run on background threads.

## License

MIT. See [LICENSE](LICENSE).
