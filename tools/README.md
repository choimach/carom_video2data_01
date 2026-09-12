# tools

Scripts that drive the pipeline over many matches. Run them with the WSL
interpreter, from the project root:

    ~/.venvs/carom/bin/python tools/screen_run.py [listing.txt]

`screen_run.py` takes a listing of `vod_id|event|title` lines and decides which
matches are worth downloading in full, by fetching the 540p rendition and trying
to calibrate it. Only a broadcast shot from above the table can be used, and
that is not something the title says: the 2023 Shanghai final is shot from the
side and cannot be calibrated at all. Screening costs a gigabyte per match
against ten for the real thing.

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
