import cv2
import copy
import gen.constants as constants
import numpy as np
from collections import Counter, OrderedDict
from env.tasks import get_task
from ai2thor.controller import Controller
import gen.utils.image_util as image_util
from gen.utils import game_util
from gen.utils.game_util import get_objects_of_type, get_obj_of_type_closest_to_obj


DEFAULT_RENDER_SETTINGS = {'renderImage': True,
                           'renderDepthImage': False,
                           'renderClassImage': False,
                           'renderObjectImage': False,
                           }

class ThorEnv(Controller):
    '''
    an extension of ai2thor.controller.Controller for ALFRED tasks
    '''
    def __init__(self, x_display=constants.X_DISPLAY,
                 player_screen_height=constants.DETECTION_SCREEN_HEIGHT,
                 player_screen_width=constants.DETECTION_SCREEN_WIDTH,
                 quality='MediumCloseFitShadows',
                 build_path=constants.BUILD_PATH):

        self.task = None
        super().__init__(quality=quality,
                         height=player_screen_height,
                         width=player_screen_width)
        self.local_executable_path = build_path
        # self.docker_enabled = True
        # self.headless = True

        # internal states
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

        print("ThorEnv started.")

    def _add_camera(self, render_settings=DEFAULT_RENDER_SETTINGS):
        # create_camera_action 0
        create_camera_action = {
            'action': 'AddThirdPartyCamera',
            'rotation': dict(x=0, y=180, z=0),
            'position': dict(x=-1.5, y=0.9007236, z=-2.25),
            'fieldOfView': 90,
        }
        super().step(**create_camera_action)
        # create_camera_action 1
        create_camera_action = {
            'action': 'AddThirdPartyCamera',
            'rotation': dict(x=0, y=360, z=0),
            'position': dict(x=-1.5, y=0.9007236, z=-2.25),
            'fieldOfView': 90,
        }
        super().step(**create_camera_action)

    def _set_third_party_camera(self, event, render_settings=DEFAULT_RENDER_SETTINGS):
        agent = event.metadata['agent']
        position = agent['position']
        rotation = agent['rotation']
        cameraHorizon = agent['cameraHorizon']+1
        cameraHorizon = 90
        # UpdateThirdPartyCamera 0
        create_camera_action = {
            'action': 'UpdateThirdPartyCamera',
            'rotation': rotation,
            'position': position,
            'thirdPartyCameraId': 0,
            'fieldOfView': cameraHorizon,
        }
        create_camera_action['rotation']["y"] -= 90
        event = super().step(**create_camera_action)
        # UpdateThirdPartyCamera 1
        create_camera_action['rotation']["y"] += 180
        create_camera_action['thirdPartyCameraId'] = 1
        event = super().step(**create_camera_action)
        return event

    def reset(self, scene_name_or_num,
              grid_size=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
              camera_y=constants.CAMERA_HEIGHT_OFFSET,
              render_image=constants.RENDER_IMAGE,
              render_depth_image=constants.RENDER_DEPTH_IMAGE,
              render_class_image=constants.RENDER_CLASS_IMAGE,
              render_object_image=constants.RENDER_OBJECT_IMAGE,
              visibility_distance=constants.VISIBILITY_DISTANCE):
        '''
        reset scene and task states
        '''
        print("Resetting ThorEnv")

        if type(scene_name_or_num) == str:
            scene_name = scene_name_or_num
        else:
            scene_name = 'FloorPlan%d' % scene_name_or_num
        super().reset(scene_name)
        event = super().step(
            action='Initialize',
            gridSize=grid_size,
            cameraY=camera_y,
            renderImage=render_image,
            renderDepthImage=render_depth_image,
            renderClassImage=render_class_image,
            renderObjectImage=render_object_image,
            visibility_distance=visibility_distance,
            makeAgentsVisible=False,
        )
        # import pdb; pdb.set_trace()

        # reset task if specified
        if self.task is not None:
            self.task.reset()

        # clear object state changes
        self.reset_states()

        self._add_camera()
        event = self._set_third_party_camera(event)

        return event

    def reset_states(self):
        '''
        clear state changes
        '''
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

    def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
        '''
        restore object locations and states
        object_poses: [{'objectName': 'Mug_a12c171b', 'position': {'x': -1.33920825, 'y': 1.72145629, 'z': 0.4894059}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'Mug_a12c171b', 'position': {'x': -1.83404362, 'y': 0.8807978, 'z': 0.4075809}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'CD_05a2c75a', 'position': {'x': -0.473243952, 'y': 0.083159484, 'z': -0.8529663}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'CD_05a2c75a', 'position': {'x': -2.2402215, 'y': 0.0391722359, 'z': 2.68283415}, 'rotation': {'x': 0.0, 'y': 180.000168, 'z': 0.0}}, {'objectName': 'CellPhone_ca4b3ad9', 'position': {'x': 1.81535876, 'y': 0.784387946, 'z': 1.36632764}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'Pencil_f507937c', 'position': {'x': -0.567746758, 'y': 0.8613946, 'z': -1.18322909}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'CreditCard_3e601b13', 'position': {'x': -0.819493532, 'y': 0.8577152, 'z': -0.836806238}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'KeyChain_0522d00d', 'position': {'x': -2.075194, 'y': 0.668328047, 'z': 0.6456865}, 'rotation': {'x': 0.0, 'y': 90.0, 'z': 0.0}}, {'objectName': 'Pillow_492a1c0b', 'position': {'x': 1.40960121, 'y': 0.918681443, 'z': 1.94519484}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'Laptop_50f3d2ae', 'position': {'x': 0.800964832, 'y': 0.781461, 'z': 0.7874603}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'Laptop_50f3d2ae', 'position': {'x': 0.800964832, 'y': 0.781461, 'z': 1.94519484}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'Book_b5ec5330', 'position': {'x': 1.00384367, 'y': 0.783036, 'z': 1.65576124}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'AlarmClock_bd50a771', 'position': {'x': -2.09172368, 'y': 0.879870236, 'z': 0.665508866}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'BasketBall_4d3c931a', 'position': {'x': 1.9761951, 'y': 0.1200004, 'z': 0.2112078}, 'rotation': {'x': 0.0, 'y': 331.0765, 'z': 0.0}}, {'objectName': 'Pen_2fb4e25b', 'position': {'x': -1.72826767, 'y': 1.35510921, 'z': 0.5303184}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'Laptop_50f3d2ae', 'position': {'x': -1.783595, 'y': 0.8719339, 'z': 0.7320865}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'BaseballBat_c36151df', 'position': {'x': 1.953, 'y': 0.659, 'z': -1.81}, 'rotation': {'x': 346.388153, 'y': 0.0, 'z': 0.0}}, {'objectName': 'Bowl_e11a5ffa', 'position': {'x': -1.83404386, 'y': 1.72700715, 'z': 0.448493421}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'Pillow_492a1c0b', 'position': {'x': 1.118, 'y': 0.867, 'z': 2.464}, 'rotation': {'x': 0.0, 'y': 0.52026695, 'z': 0.0}}, {'objectName': 'KeyChain_0522d00d', 'position': {'x': -2.292484, 'y': 0.03971261, 'z': 2.52258325}, 'rotation': {'x': 0.0, 'y': 180.000168, 'z': 0.0}}, {'objectName': 'CreditCard_3e601b13', 'position': {'x': -1.83404362, 'y': 0.8827938, 'z': 0.530318558}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'CellPhone_ca4b3ad9', 'position': {'x': -0.735577941, 'y': 0.858646154, 'z': -1.01001763}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'CD_05a2c75a', 'position': {'x': -0.748, 'y': 0.202964082, 'z': -0.338}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'Pencil_f507937c', 'position': {'x': -1.1798991, 'y': 1.72713172, 'z': 0.4894059}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}, {'objectName': 'Mug_a12c171b', 'position': {'x': -0.65166235, 'y': 0.8557192, 'z': -0.663594842}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}]
        object_toggles: [{'isOn': False, 'objectType': 'DeskLamp'}]
        '''
        super().step(
            action='Initialize',
            gridSize=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
            cameraY=constants.CAMERA_HEIGHT_OFFSET,
            renderImage=constants.RENDER_IMAGE,
            renderDepthImage=constants.RENDER_DEPTH_IMAGE,
            renderClassImage=constants.RENDER_CLASS_IMAGE,
            renderObjectImage=constants.RENDER_OBJECT_IMAGE,
            visibility_distance=constants.VISIBILITY_DISTANCE,
            makeAgentsVisible=False,
        )
        if len(object_toggles) > 0:
            # import pdb; pdb.set_trace()
            # old
            # super().step((dict(action='SetObjectToggles', objectToggles=object_toggles)))
            # new
            # event = super().step(action='SetObjectStates', SetObjectStates={'objectType': 'DeskLamp', 'stateChange': 'toggleable', 'isToggled': False})
            for object_toggle in object_toggles:
                object_toggle['isToggled'] = object_toggle['isOn']
                object_toggle['isOpen'] = object_toggle['isOn']
                object_toggle['stateChange'] = "toggleable"
                event = super().step(action='SetObjectStates', SetObjectStates=object_toggle)
                print("object_toggles: ")
                print(event.metadata['lastActionSuccess'])
                print(event.metadata['errorMessage'])

        if dirty_and_empty:
            # new
            event = super().step(action='SetObjectStates', SetObjectStates={"stateChange": "dirtyable"})
            print("SetObjectStates: ")
            print(event.metadata['lastActionSuccess'])
            print(event.metadata['errorMessage'])
            event = super().step(action='SetObjectStates', SetObjectStates={"stateChange": "canFillWithLiquid"})
            print("SetObjectStates: ")
            print(event.metadata['lastActionSuccess'])
            print(event.metadata['errorMessage'])
            # old
            # super().step(dict(action='SetStateOfAllObjects',
            #                    StateChange="CanBeDirty",
            #                    forceAction=True))
            # super().step(dict(action='SetStateOfAllObjects',
            #                    StateChange="CanBeFilled",
            #                    forceAction=False))
        event = super().step(action='SetObjectPoses', objectPoses=object_poses)
        # print(event.metadata['objects'])
        # import pdb; pdb.set_trace()
        print("SetObjectPoses: ")
        print(event.metadata['lastActionSuccess'])
        print(event.metadata['errorMessage'])

    def set_task(self, traj, args, reward_type='sparse', max_episode_length=2000):
        '''
        set the current task type (one of 7 tasks)
        '''
        task_type = traj['task_type']
        self.task = get_task(task_type, traj, self, args, reward_type=reward_type, max_episode_length=max_episode_length)

    def step(self, smooth_nav=False, **action_args):
        '''
        overrides ai2thor.controller.Controller.step() for smooth navigation and goal_condition updates
        '''
        action = action_args["action"]
        if smooth_nav:
            if "MoveAhead" in action:
                self.smooth_move_ahead(**action_args)
            elif "Rotate" in action:
                self.smooth_rotate(**action_args)
            elif "Look" in action:
                self.smooth_look(**action_args)
            else:
                super().step(**action_args)
        else:
            if "LookUp" in action:
                self.look_angle(-constants.AGENT_HORIZON_ADJ)
            elif "LookDown" in action:
                self.look_angle(constants.AGENT_HORIZON_ADJ)
            else:
                super().step(**action_args)

        event = self.update_states(**action_args)
        if not event.metadata['lastActionSuccess']:
            print("step: ")
            print(event.metadata['errorMessage'])
        event = self._set_third_party_camera(event)
        self.check_post_conditions(**action_args)
        return event

    def check_post_conditions(self, **action):
        '''
        handle special action post-conditions
        '''
        if action["action"] == 'ToggleObjectOn':
            self.check_clean(action['objectId'])

    def update_states(self, **action):
        '''
        extra updates to metadata after step
        '''
        # add 'cleaned' to all object that were washed in the sink
        event = self.last_event
        if event.metadata['lastActionSuccess']:
            # clean
            if action['action'] == 'ToggleObjectOn' and "Faucet" in action['objectId']:
                sink_basin = get_obj_of_type_closest_to_obj('SinkBasin', action['objectId'], event.metadata)
                cleaned_object_ids = sink_basin['receptacleObjectIds']
                self.cleaned_objects = self.cleaned_objects | set(cleaned_object_ids) if cleaned_object_ids is not None else set()
            # heat
            if action['action'] == 'ToggleObjectOn' and "Microwave" in action['objectId']:
                microwave = get_objects_of_type('Microwave', event.metadata)[0]
                heated_object_ids = microwave['receptacleObjectIds']
                self.heated_objects = self.heated_objects | set(heated_object_ids) if heated_object_ids is not None else set()
            # cool
            if action['action'] == 'CloseObject' and "Fridge" in action['objectId']:
                fridge = get_objects_of_type('Fridge', event.metadata)[0]
                cooled_object_ids = fridge['receptacleObjectIds']
                self.cooled_objects = self.cooled_objects | set(cooled_object_ids) if cooled_object_ids is not None else set()

        return event

    def get_transition_reward(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for transition_reward")
        else:
            return self.task.transition_reward(self.last_event)

    def get_goal_satisfied(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_satisfied(self.last_event)

    def get_goal_conditions_met(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_conditions_met(self.last_event)

    def get_subgoal_idx(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for subgoal_idx")
        else:
            return self.task.get_subgoal_idx()

    def noop(self):
        '''
        do nothing
        '''
        super().step(action='Pass')

    def smooth_move_ahead(self, render_settings=None, **action):
        '''
        smoother MoveAhead
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        smoothing_factor = constants.RECORD_SMOOTHING_FACTOR
        new_action = copy.deepcopy(action)
        new_action['moveMagnitude'] = constants.AGENT_STEP_SIZE / smoothing_factor

        new_action['renderImage'] = render_settings['renderImage']
        new_action['renderClassImage'] = render_settings['renderClassImage']
        new_action['renderObjectImage'] = render_settings['renderObjectImage']
        new_action['renderDepthImage'] = render_settings['renderDepthImage']

        events = []
        for xx in range(smoothing_factor - 1):
            action.update(new_action)
            event = super().step(**action)
            if event.metadata['lastActionSuccess']:
                event = self._set_third_party_camera(event)
                events.append(event)
        action.update(new_action)
        event = super().step(**action)
        if event.metadata['lastActionSuccess']:
            event = self._set_third_party_camera(event)
            events.append(event)
        else:
            print(event.metadata['errorMessage'])
            return [event]
        return events

    def smooth_rotate(self, render_settings=None, **action):
        '''
        smoother RotateLeft and RotateRight
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        if action['action'] == 'RotateLeft':
            end_rotation = (start_rotation - 90)
        else:
            end_rotation = (start_rotation + 90)

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                    'tempRenderChange': True,
                    'renderNormalsImage': False,
                    'renderImage': render_settings['renderImage'],
                    'renderClassImage': render_settings['renderClassImage'],
                    'renderObjectImage': render_settings['renderObjectImage'],
                    'renderDepthImage': render_settings['renderDepthImage'],
                }
                event = super().step(**teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                }
                event = super().step(**teleport_action)

            if event.metadata['lastActionSuccess']:
                event = self._set_third_party_camera(event)
                events.append(event)
        return events

    def smooth_look(self, render_settings=None, **action):
        '''
        smoother LookUp and LookDown
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + constants.AGENT_HORIZON_ADJ * (1 - 2 * int(action['action'] == 'LookUp'))
        position = event.metadata['agent']['position']

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': rotation,
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                    'tempRenderChange': True,
                    'renderNormalsImage': False,
                    'renderImage': render_settings['renderImage'],
                    'renderClassImage': render_settings['renderClassImage'],
                    'renderObjectImage': render_settings['renderObjectImage'],
                    'renderDepthImage': render_settings['renderDepthImage'],
                }
                event = super().step(**teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': rotation,
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                }
                event = super().step(**teleport_action)

            if event.metadata['lastActionSuccess']:
                event = self._set_third_party_camera(event)
                events.append(event)
        return events

    def look_angle(self, angle, render_settings=None):
        '''
        look at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + angle
        position = event.metadata['agent']['position']

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': rotation,
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': np.round(end_horizon, 3),
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'renderImage': render_settings['renderImage'],
            'renderClassImage': render_settings['renderClassImage'],
            'renderObjectImage': render_settings['renderObjectImage'],
            'renderDepthImage': render_settings['renderDepthImage'],
        }
        event = super().step(**teleport_action)
        event = self._set_third_party_camera(event)
        return event

    def rotate_angle(self, angle, render_settings=None):
        '''
        rotate at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        end_rotation = start_rotation + angle

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': np.round(end_rotation, 3),
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': horizon,
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'renderImage': render_settings['renderImage'],
            'renderClassImage': render_settings['renderClassImage'],
            'renderObjectImage': render_settings['renderObjectImage'],
            'renderDepthImage': render_settings['renderDepthImage'],
        }
        event = super().step(**teleport_action)
        event = self._set_third_party_camera(event)
        return event

    def to_thor_api_exec(self, action, object_id="", smooth_nav=False):
        # TODO: parametrized navigation commands

        if "RotateLeft" in action:
            action = dict(action="RotateLeft",
                          forceAction=True)
            event = self.step(smooth_nav=smooth_nav, **action)
        elif "RotateRight" in action:
            action = dict(action="RotateRight",
                          forceAction=True)
            event = self.step(smooth_nav=smooth_nav, **action)
        elif "MoveAhead" in action:
            action = dict(action="MoveAhead",
                          forceAction=True)
            event = self.step(smooth_nav=smooth_nav, **action)
        elif "LookUp" in action:
            action = dict(action="LookUp",
                          forceAction=True)
            event = self.step(smooth_nav=smooth_nav, **action)
        elif "LookDown" in action:
            action = dict(action="LookDown",
                          forceAction=True)
            event = self.step(smooth_nav=smooth_nav, **action)
        elif "OpenObject" in action:
            action = dict(action="OpenObject",
                          objectId=object_id,
                          moveMagnitude=1.0)
            event = self.step(**action)
        elif "CloseObject" in action:
            action = dict(action="CloseObject",
                          objectId=object_id,
                          forceAction=True)
            event = self.step(**action)
        elif "PickupObject" in action:
            action = dict(action="PickupObject",
                          objectId=object_id)
            event = self.step(**action)
        elif "PutObject" in action:
            inventory_object_id = self.last_event.metadata['inventoryObjects'][0]['objectId']
            action = dict(action="PutObject",
                          objectId=inventory_object_id,
                          receptacleObjectId=object_id,
                          forceAction=True,
                          placeStationary=True)
            event = self.step(**action)
        elif "ToggleObjectOn" in action:
            action = dict(action="ToggleObjectOn",
                          objectId=object_id)
            event = self.step(**action)

        elif "ToggleObjectOff" in action:
            action = dict(action="ToggleObjectOff",
                          objectId=object_id)
            event = self.step(**action)
        elif "SliceObject" in action:
            # check if agent is holding knife in hand
            inventory_objects = self.last_event.metadata['inventoryObjects']
            if len(inventory_objects) == 0 or 'Knife' not in inventory_objects[0]['objectType']:
                raise Exception("Agent should be holding a knife before slicing.")

            action = dict(action="SliceObject",
                          objectId=object_id)
            event = self.step(**action)
        else:
            raise Exception("Invalid action. Conversion to THOR API failed! (action='" + str(action) + "')")

        return event, action

    def check_clean(self, object_id):
        '''
        Handle special case when Faucet is toggled on.
        In this case, we need to execute a `CleanAction` in the simulator on every object in the corresponding
        basin. This is to clean everything in the sink rather than just things touching the stream.
        '''
        event = self.last_event
        if event.metadata['lastActionSuccess'] and 'Faucet' in object_id:
            # Need to delay one frame to let `isDirty` update on stream-affected.
            event = self.step(**{'action': 'Pass'})
            sink_basin_obj = game_util.get_obj_of_type_closest_to_obj("SinkBasin", object_id, event.metadata)
            for in_sink_obj_id in sink_basin_obj['receptacleObjectIds']:
                if (game_util.get_object(in_sink_obj_id, event.metadata)['dirtyable']
                        and game_util.get_object(in_sink_obj_id, event.metadata)['isDirty']):
                    event = self.step(**{'action': 'CleanObject', 'objectId': in_sink_obj_id})
        return event

    def prune_by_any_interaction(self, instances_ids):
        '''
        ignores any object that is not interactable in anyway
        '''
        pruned_instance_ids = []
        for obj in self.last_event.metadata['objects']:
            obj_id = obj['objectId']
            if obj_id in instances_ids:
                if obj['pickupable'] or obj['receptacle'] or obj['openable'] or obj['toggleable'] or obj['sliceable']:
                    pruned_instance_ids.append(obj_id)

        ordered_instance_ids = [id for id in instances_ids if id in pruned_instance_ids]
        return ordered_instance_ids

    def va_interact(self, action, interact_mask=None, smooth_nav=True, mask_px_sample=1, debug=False):
        '''
        interact mask based action call
        '''

        all_ids = []

        if type(interact_mask) is str and interact_mask == "NULL":
            raise Exception("NULL mask.")
        elif interact_mask is not None:
            # ground-truth instance segmentation mask from THOR
            instance_segs = np.array(self.last_event.instance_segmentation_frame)
            color_to_object_id = self.last_event.color_to_object_id

            # get object_id for each 1-pixel in the interact_mask
            nz_rows, nz_cols = np.nonzero(interact_mask)
            instance_counter = Counter()
            for i in range(0, len(nz_rows), mask_px_sample):
                x, y = nz_rows[i], nz_cols[i]
                instance = tuple(instance_segs[x, y])
                instance_counter[instance] += 1
            if debug:
                print("action_box", "instance_counter", instance_counter)

            # iou scores for all instances
            iou_scores = {}
            for color_id, intersection_count in instance_counter.most_common():
                union_count = np.sum(np.logical_or(np.all(instance_segs == color_id, axis=2), interact_mask.astype(bool)))
                iou_scores[color_id] = intersection_count / float(union_count)
            iou_sorted_instance_ids = list(OrderedDict(sorted(iou_scores.items(), key=lambda x: x[1], reverse=True)))

            # get the most common object ids ignoring the object-in-hand
            inv_obj = self.last_event.metadata['inventoryObjects'][0]['objectId'] \
                if len(self.last_event.metadata['inventoryObjects']) > 0 else None
            all_ids = [color_to_object_id[color_id] for color_id in iou_sorted_instance_ids
                       if color_id in color_to_object_id and color_to_object_id[color_id] != inv_obj]

            # print all ids
            if debug:
                print("action_box", "all_ids", all_ids)

            # print instance_ids
            instance_ids = [inst_id for inst_id in all_ids if inst_id is not None]
            if debug:
                print("action_box", "instance_ids", instance_ids)

            # prune invalid instances like floors, walls, etc.
            instance_ids = self.prune_by_any_interaction(instance_ids)

            # cv2 imshows to show image, segmentation mask, interact mask
            if debug:
                print("action_box", "instance_ids", instance_ids)
                instance_seg = copy.copy(instance_segs)
                instance_seg[:, :, :] = interact_mask[:, :, np.newaxis] == 1
                instance_seg *= 255

                cv2.imshow('seg', instance_segs)
                cv2.imshow('mask', instance_seg)
                cv2.imshow('full', self.last_event.frame[:,:,::-1])
                cv2.waitKey(0)

            if len(instance_ids) == 0:
                err = "Bad interact mask. Couldn't locate target object"
                success = False
                return success, None, None, err, None

            target_instance_id = instance_ids[0]
        else:
            target_instance_id = ""

        if debug:
            print("taking action: " + str(action) + " on target_instance_id " + str(target_instance_id))
        try:
            event, api_action = self.to_thor_api_exec(action, target_instance_id, smooth_nav)
        except Exception as err:
            success = False
            return success, None, None, err, None

        if not event.metadata['lastActionSuccess']:
            if interact_mask is not None and debug:
                print("Failed to execute action!", action, target_instance_id)
                print("all_ids inside BBox: " + str(all_ids))
                instance_seg = copy.copy(instance_segs)
                instance_seg[:, :, :] = interact_mask[:, :, np.newaxis] == 1
                cv2.imshow('seg', instance_segs)
                cv2.imshow('mask', instance_seg)
                cv2.imshow('full', self.last_event.frame[:,:,::-1])
                cv2.waitKey(0)
                print(event.metadata['errorMessage'])
            success = False
            return success, event, target_instance_id, event.metadata['errorMessage'], api_action

        success = True
        return success, event, target_instance_id, '', api_action

    @staticmethod
    def bbox_to_mask(bbox):
        return image_util.bbox_to_mask(bbox)

    @staticmethod
    def point_to_mask(point):
        return image_util.point_to_mask(point)

    @staticmethod
    def decompress_mask(compressed_mask):
        return image_util.decompress_mask(compressed_mask)
