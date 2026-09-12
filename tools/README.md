# tools

Scripts that drive the pipeline over many matches. Run them with the WSL
interpreter, from the project root:

    ~/.venvs/carom/bin/python tools/screen_run.py [listing.txt]

`screen_run.py` takes a listing of `vod_id|event|title` lines and decides which
matches are worth downloading in full. Only a broadcast shot from above the
table can be used, and that is not something the title says: the 2023 Shanghai
final is shot from the side and cannot be calibrated at all.

It judges a match by pulling a dozen-odd HLS segments from across it - tens of
megabytes and half a minute, against ten gigabytes and half an hour for the
match itself - and calibrating a frame from each. The segments come at the
broadcast's own resolution, which matters: downloading the 540p rendition
instead is both slower and wrong, because a camera pulled well back leaves its
diamonds three pixels across at that size and the match reads as having no
overhead camera when it plainly has one.

Results accumulate in `data/_screening.json` as each match is judged, so the run
can be stopped and resumed.

`pick_matches.py` chooses what to screen next, from a catalogue written by

    yt-dlp --flat-playlist --print '%(id)s|%(title)s' \
        https://ch.sooplive.co.kr/afbilliards1/vods > tools/catalogue.txt

It orders by recency - the VOD id rises with time, so no date lookup is needed -
and takes one match per event before repeating, so a single unusable production
cannot sink the whole sample. Matches already in `data/_screening.json` are
skipped, so picking and screening can alternate without repeating work.

Recency matters because the production changed: the 2023 coverage under the old
AfreecaTV branding is shot from the side of the table and cannot be calibrated
at all, while the 2026 SOOP coverage is overhead.
