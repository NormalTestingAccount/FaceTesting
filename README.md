Most of this shit is useless testing shenanigans, except for:


- main.py | obviously the main file to run
- post_process.py | handles all of the facial recognition processing-associated shite
- recognition.py | handles the second process (facial encoding and frame processing)

all in all, it works OK. it would be beneficial to do the following, however:

- implement face rotation filtering (the landmarks are already provided for ~~sanity~~ convenience)
- implement face size filtering (really small faces don't get recognized well. who would have thought)
- implement face edge-of-frame filtering (cut-off faces don't get recognized well. another no-brainer)
- implement blurry face filtering (for obvious reasons.)

- IMPORTANT: MAKE AI NOT RACIST (do NOT ask it to recognize brown people right now.)

- oh, and optimize RetinaFace pre+post-processing (its slow as fuck rn)
    - maybe see what operations are static (same every runthrough) and move those to front?
    - alternatively, you could also try consolidating every numpy operation into a function, and trying to make that into an ONNX graph. Probably the most batshit idea but also would be the fastest if it worked.


forgot to mention:
- supervision works now for no reason, no edit required (reinstall if you have it already)
- dlib_bin is a lifesaver, but it might not work with cuda well.
- was running at ~15-20fps on a laptop with decent CPU and ☠️ gpu, don't know how well it'll run on other devices.
- God Bless America