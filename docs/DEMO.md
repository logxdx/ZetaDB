# Demo library

A small, good-looking library for trying ZetaDB and for the screenshots in the README: 10 to 20
of each kind of thing ZetaDB searches.

| | Count | What |
| --- | ---: | --- |
| Photos | 16 | landscapes, cities, wildlife, a neon sign — from Unsplash |
| Documents | 10 | receipt, boarding pass, whiteboard, chart, slide, terminal, code, recipe, ticket, handwritten list — rendered locally, all invented |
| Notes | 12 | trip plans, meeting notes with to-dos, a bug report, recipes — in [`notes.json`](../scripts/demo/notes.json) |
| Audio | 12 | speeches (Kennedy, Roosevelt, Apollo 11), piano, cello, ragtime, rain, birdsong, waves, fire, bells, fireworks |
| Video | 11 | two Blender open movies (*Sintel*, *Big Buck Bunny*), a rocket launch, an aurora, hummingbirds at 1500 fps, a cat, a waterfall, snow, a city street |

The two long films and Kennedy's 17-minute speech are what make **moment search** worth
showing: *"why we go to the moon"* opens the speech at 8:20, *"a dragon"* opens *Sintel* at 5:20.

## Build it

With ZetaDB running (`uv run -m server.app`):

```sh
uv run scripts/demo/fetch.py ~/zetadb-demo                                          # ~260 MB
uv run --no-project --with playwright scripts/demo/make_docs.py ~/zetadb-demo/docs  # the 10 documents
uv run scripts/demo/load.py ~/zetadb-demo                                           # upload + titles
```

`fetch.py` skips files it already has and backs off when Wikimedia rate-limits it. Ingestion
then runs in the background: on a 6 GB laptop GPU the whole library is searchable in about five
minutes.

Use a separate data folder for the demo (a worktree, or a checkout whose `config.toml` points
`paths` elsewhere) if you do not want it mixed into your own library.

## Sources and licences

The documents and notes are invented for this demo: the café, airline, band, company and people
in them do not exist. Everything else is listed below. Unsplash photos are under the
[Unsplash License](https://unsplash.com/license); public-domain and CC0 works need no
attribution but are credited anyway; CC BY and CC BY-SA works are credited as their licences
require, and none were changed beyond scaling and transcoding.

### Photos

| Photo | Photographer | Licence |
| --- | --- | --- |
| [Gull at sunset · Kevin Grieve](https://unsplash.com/photos/BO606huIYiU) | Kevin Grieve | Unsplash License |
| [Plum blossom · Pic Studio](https://unsplash.com/photos/HAv6QQch0S0) | Pic Studio | Unsplash License |
| [Downtown at golden hour · Tonny Huang](https://unsplash.com/photos/NkIGNyej0Ww) | Tonny Huang | Unsplash License |
| [San Francisco skyline · Kellen Riggin](https://unsplash.com/photos/pKEGyLcl1ac) | Kellen Riggin | Unsplash License |
| [Bay Bridge at dusk · Kellen Riggin](https://unsplash.com/photos/caB-1hKvt1Y) | Kellen Riggin | Unsplash License |
| [Deer below Mount Rainier · Emma Swoboda](https://unsplash.com/photos/DXEhDakyt8E) | Emma Swoboda | Unsplash License |
| [Clouds catching the sun · Tom Barrett](https://unsplash.com/photos/hgGplX3PFBg) | Tom Barrett | Unsplash License |
| [Sea arch, Faroe Islands · Sebastian Boring](https://unsplash.com/photos/8zD7rs8UpxU) | Sebastian Boring | Unsplash License |
| [Snowy peaks · Simon Fitall](https://unsplash.com/photos/5ditmO9_ae0) | Simon Fitall | Unsplash License |
| [Turquoise car on a coastal road · Joshua S](https://unsplash.com/photos/dmUnpAUakNM) | Joshua S | Unsplash License |
| [Sunlight through a giant fig · Jeremy Bishop](https://unsplash.com/photos/EwKXn5CapA4) | Jeremy Bishop | Unsplash License |
| [Shoreline from above · Shifaaz Shamoon](https://unsplash.com/photos/sLAk1guBG90) | Shifaaz Shamoon | Unsplash License |
| [Lake in the pines · Jeremy Brady](https://unsplash.com/photos/vK26p2SFX3E) | Jeremy Brady | Unsplash License |
| [“be kind.” neon · Itsiken](https://unsplash.com/photos/_n424c0wmEY) | Itsiken | Unsplash License |
| [Dolomites at dusk · Massimiliano Morosinotto](https://unsplash.com/photos/uJIhAvqFJxg) | Massimiliano Morosinotto | Unsplash License |
| [Mountain road in autumn · Ryan Hoffman](https://unsplash.com/photos/Pqk44XmCgpk) | Ryan Hoffman | Unsplash License |

### Audio and video

| Title | Author | Licence |
| --- | --- | --- |
| [JFK — “We choose to go to the Moon” (1962)](https://commons.wikimedia.org/wiki/File:Jfk_rice_university_we_choose_to_go_to_the_moon.ogg) | John F. Kennedy | Public domain |
| [Apollo 11 — “One small step” (1969)](https://commons.wikimedia.org/wiki/File:Armstrong_Small_Step.ogg) | Neil Armstrong / NASA | Public domain |
| [FDR — First inaugural address (1933)](https://commons.wikimedia.org/wiki/File:First_Inauguration_of_FDR_-_Fear_Itself_Excerpt.ogg) | Franklin D. Roosevelt | Public domain |
| [Beethoven — Moonlight Sonata, Adagio](https://commons.wikimedia.org/wiki/File:Ludwig_van_Beethoven_-_sonata_no._14_in_c_sharp_minor_%27moonlight%27%2C_op._27_no._2_-_i._adagio_sostenuto.ogg) | Musopen | Public domain |
| [Scott Joplin — Maple Leaf Rag (1916)](https://commons.wikimedia.org/wiki/File:Maple_leaf_rag_-_played_by_Scott_Joplin_1916_V2.ogg) | Scott Joplin | Public domain |
| [Bach — Cello Suite No. 1, Prelude](https://commons.wikimedia.org/wiki/File:Bach_Cello_Suite_1_Prelude_%28BWV_1007%29_Played_by_Chris.ogg) | Fopseh | CC0 |
| [Rain and thunder](https://commons.wikimedia.org/wiki/File:Rain_and_thunder_%281%29.ogg) | ezwa | Public domain |
| [Birdsong, Bourne Woods](https://commons.wikimedia.org/wiki/File:Bourne_Woods_Birdsong_2020-04-27_0751.mp3) | Robert EA Harvey | CC BY-SA 4.0 |
| [Waves on the shore](https://commons.wikimedia.org/wiki/File:Oceanwavescrushing.ogg) | Luftrum | CC BY 3.0 |
| [Wood fire crackling](https://commons.wikimedia.org/wiki/File:Bones_breaking_wood_fire_ice_crackling.ogg) | stephan | Public domain |
| [New Year's fireworks from the street](https://commons.wikimedia.org/wiki/File:Silvester_fireworks_from_the_street_02_cool_stuff.ogg) | stephan | Public domain |
| [Church bells, Leverkusen](https://commons.wikimedia.org/wiki/File:Church_bells_-_Leverkusen%2C_2007.oga) | natalie | Public domain |
| [Big Buck Bunny](https://commons.wikimedia.org/wiki/File:Big_Buck_Bunny_4K.webm) | Blender Foundation | CC BY 3.0 |
| [Sintel](https://commons.wikimedia.org/wiki/File:Sintel_movie_4K.webm) | Blender Foundation (Durian Open Movie project) | CC BY 3.0 |
| [Ocean waves](https://commons.wikimedia.org/wiki/File:Ocean_surface_waves_05.ogv) | Mostafameraji | CC0 |
| [Watagataki Falls](https://commons.wikimedia.org/wiki/File:Watagataki_Falls_%28Video%29.webm) | Yoshi Canopus | CC0 |
| [Aurora over Alaska](https://commons.wikimedia.org/wiki/File:Alaskan_Aurora_Time_Lapse_%28SVS15077%29.webm) | NASA Scientific Visualization Studio | Public domain |
| [Artemis I launch, slow motion](https://commons.wikimedia.org/wiki/File:Artemis_I_Launch_from_NASA_Causeway_%28Slo_Mo%29_%28GX010506%29.webm) | NASA Marshall Space Flight Center | Public domain |
| [Hummingbirds at 1500 fps](https://commons.wikimedia.org/wiki/File:Humming_birds_feeding_at_1500fps.webm) | HopefullyUsefulContribution | CC BY-SA 4.0 |
| [Underwater fireworks, Kamakura](https://commons.wikimedia.org/wiki/File:Kamakura_underwater_fireworks_2.webm) | Yoshi Canopus | CC0 |
| [Cat meets fish tank](https://commons.wikimedia.org/wiki/File:Cat_reacting_to_a_fish_tank.webm) | Danmation | CC BY 3.0 |
| [Snowfall](https://commons.wikimedia.org/wiki/File:Snowfall.webm) | Cody Logan | CC BY 3.0 |
| [City street time-lapse](https://commons.wikimedia.org/wiki/File:City_street_time_lapse.webm) | Editor (YouTube) | CC BY 3.0 |
