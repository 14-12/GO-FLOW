Hi Hela,

Thanks for the full recordings. We ran v1 on all 18 hours and tried three v2 approaches.

The main finding is that the 10 % careless "normal" runs contain long jams and blocked junctions, caused by crashes elsewhere in town. Our agent must alarm only on accidents, never on jams. Rules built on stopped cars therefore fail.

New accident-specific signals (a crash moment, and cars driving around a stopped car) stay quiet in jams, at about 0.2 false alarms per camera-hour. But the training runs hold only one clearly visible crash, so we cannot validate anything yet.

Could you send us:

1. **`carla-yolo26n-t05.pt`**. Its police_lights, ambulance_lights and truck classes give the strongest accident signs.

2. **10–20 more crashes clearly visible to a fixed camera:**
   - in new runs (new seeds) with the normal 10 % careless share;
   - wreck within about 30 m of the camera and fully in the picture, not at the frame edge;
   - impact happening on screen, with at least 2 min of video before it and 3 min after;
   - a mix of ring-road and junction cameras and of times of day.

   For each crash, please give us the same files as before (`run.json`, `incidents.json` with "visible", `<camera>.frames.csv`). Please also add:
   - the on-screen impact second;
   - the lane;
   - which vehicles are the wrecks.

   Please also say which runs we may train on and which stay held out.

3. **A question.** On screen, the cars in the 003054 crash slow from driving to standing over about 2–3 s, so the impact looks like hard braking. Is that the viewer's position interpolation? Can clips be recorded without it?

Thanks!
