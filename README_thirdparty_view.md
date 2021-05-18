# Agent Camera tp 3 times (ai2thor==2.1.0)
get_third_party_frame
set_hand_object_hide
https://github.com/allenai/ai2thor/issues/537
## get raw_images_2, raw_images_1
```
python scripts/augment_trajectories_third_party_camera_frames210.py --data_path ../data/full_2.1.0/ --num_threads 10 --smooth_nav --time_delays
```

## get exploration_meta_1, sgg_meta_1
```
python scripts/augment_meta_data_trajectories_third_party_camera_frames.py --data_path ../data/full_2.1.0/ --num_threads 10 --smooth_nav --time_delays
```

---
---

# Add Camera (ai2thor==2.5.4)
```
controller.step('AddThirdPartyCamera',
    rotation=dict(x=0, y=90, z=0), 
    position=dict(x=-1.25, y=1.0, z=-1.0),
    fieldOfView=60)
```

```
controller.step('UpdateThirdPartyCamera',
    thirdPartyCameraId=0, # id is available in the metadata response
    rotation=dict(x=0, y=90, z=0),
    position=dict(x=-1.25, y=1.0, z=-1.5)
    )
```

## get Camera frame
```
event.metadata['thirdPartyCameras']
```
{'thirdPartyCameraId': 0, 'position': {'x': -0.75, 'y': 0.9101201295852661, 'z': 0.25}, 'rotation': {'x': 0.0, 'y': 270.0, 'z': 0.0}, 'fieldOfView': 90.0}
{'thirdPartyCameraId': 1, 'position': {'x': -0.75, 'y': 0.9101201295852661, 'z': 0.25}, 'rotation': {'x': -0.0, 'y': 90.0, 'z': 0.0}, 'fieldOfView': 90.0}
```
event.third_party_camera_frames
# len(event.third_party_camera_frames) = 2
```

## metadata['agent']
event.metadata['agent']
```
{{'name': 'agent', 'position': {'x': -0.75, 'y': 0.9101201295852661, 'z': 0.25}, 'rotation': {'x': -0.0, 'y': 0.0, 'z': 0.0}, 'cameraHorizon': 30.000003814697266, 'isStanding': True, 'inHighFrictionArea': False}
```


---
---
