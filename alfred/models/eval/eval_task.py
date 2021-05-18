import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
from eval import Eval
from env.thor_env import ThorEnv
from models.utils.save_video import images_to_video

class EvalTask(Eval):
    '''
    evaluate overall task performance
    '''

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures, results):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()

        while True:
            if task_queue.qsize() == 0:
                break

            task = task_queue.get()

            try:
                traj = model.load_task_json(task)
                r_idx = task['repeat_idx']
                print("Evaluating: %s" % (traj['root']))
                print("No. of trajectories left: %d" % (task_queue.qsize()))
                try:
                    if model.semantic_graph_implement.use_exploration_frame_feats:
                        meta_datas = cls.explore_scene(env, model)
                        model.semantic_graph_implement.update_exploration_data_to_global_graph(
                            meta_datas,
                            0
                        )
                except Exception as e:
                    pass
                cls.evaluate(env, model, r_idx, resnet, traj, args, lock, successes, failures, results)
                try:
                    model.finish_of_episode()
                except Exception as e:
                    print(e)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()


    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures, results):
        # reset model
        model.reset()

        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # extract language features
        feat = model.featurize([traj_data], load_mask=False)

        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        images, depths, list_actions = [], [], []
        fail_reason = ''
        done, success = False, False
        fails = 0
        t = 0
        reward = 0
        while not done:
            # break if max_steps reached
            if t >= args.max_steps:
                break

            # extract visual features
            # env: {'request_queue': <queue.Queue object at 0x7f67452990b8>, 'response_queue': <queue.Queue object at 0x7f673d062438>, 'receptacle_nearest_pivot_points': {}, 'server': <ai2thor.server.Server object at 0x7f673d062518>, 'unity_pid': 12716, 'docker_enabled': False, 'container_id': None, 'local_executable_path': None, 'last_event': <ai2thor.server.Event object at 0x7f671419f8d0>, 'server_thread': <Thread(Thread-1, started daemon 140078244427520)>, 'killing_unity': False, 'quality': 'MediumCloseFitShadows', 'lock_file': <_io.TextIOWrapper name='/root/.ai2thor/releases/thor-201909061227-Linux64/.lock' mode='w' encoding='ANSI_X3.4-1968'>, 'fullscreen': False, 'headless': False, 'task': <env.tasks.PickHeatThenPlaceInRecepTask object at 0x7f671419f898>, 'cleaned_objects': set(), 'cooled_objects': set(), 'heated_objects': set(), 'last_action': {'action': 'TeleportFull', 'horizon': 30, 'rotateOnTeleport': True, 'rotation': {'y': 180}, 'x': -2.25, 'y': 0.8995012, 'z': 2.5, 'sequenceId': 4}}
            # env: dict_keys(['request_queue', 'response_queue', 'receptacle_nearest_pivot_points', 'server', 'unity_pid', 'docker_enabled', 'container_id', 'local_executable_path', 'last_event', 'server_thread', 'killing_unity', 'quality', 'lock_file', 'fullscreen', 'headless', 'task', 'cleaned_objects', 'cooled_objects', 'heated_objects', 'last_action'])
            # env.last_event: dict_keys(['metadata', 'screen_width', 'screen_height', 'frame', 'depth_frame', 'normals_frame', 'flow_frame', 'color_to_object_id', 'object_id_to_color', 'instance_detections2D', 'instance_masks', 'class_masks', 'instance_segmentation_frame', 'class_segmentation_frame', 'class_detections2D', 'third_party_camera_frames', 'third_party_class_segmentation_frames', 'third_party_instance_segmentation_frames', 'third_party_depth_frames', 'third_party_normals_frames', 'third_party_flows_frames', 'events'])
            image = np.uint8(env.last_event.frame)
            curr_image = Image.fromarray(image)
            feat['frames'] = resnet.featurize([curr_image], batch=1).unsqueeze(0)
            curr_instance = Image.fromarray(np.uint8(env.last_event.instance_segmentation_frame))
            feat['frames_instance'] = resnet.featurize([curr_instance], batch=1).unsqueeze(0)
            curr_depth_image = env.last_event.depth_frame * (255 / 10000)
            curr_depth_image = curr_depth_image.astype(np.uint8)
            feat['frames_depth'] = curr_depth_image
            feat['all_meta_datas'] = cls.get_meta_datas(cls, env, resnet)
            # forward model
            m_out = model.step(feat)
            m_pred = model.extract_preds(m_out, [traj_data], feat, clean_special_tokens=False)
            m_pred = list(m_pred.values())[0]

            # check if <<stop>> was predicted
            if m_pred['action_low'] == cls.STOP_TOKEN:
                print("\tpredicted STOP")
                break

            # get action and mask
            action, mask = m_pred['action_low'], m_pred['action_low_mask'][0]
            mask = np.squeeze(mask, axis=0) if model.has_interaction(action) else None

            # print action
            if args.debug:
                print(action)
            # print(action)

            # use predicted action and mask (if available) to interact with the env
            t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            if not t_success:
                fails += 1
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    fail_reason = err
                    break

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1
            images.append(image)
            depths.append(curr_depth_image)
            list_actions.append(action)
        # check if goal was satisfied
        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            print("Goal Reached")
            success = True
            video_name = "S_"
        else:
            video_name = "F_"
        video_name += traj_data['task_type'] + '_' + traj_data['task_id'] + str(traj_data['repeat_idx'])
        images_to_video(model.args.dout, video_name, images, depths, list_actions, goal_instr, fail_reason)


        # goal_conditions
        pcs = env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        lock.acquire()
        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(r_idx),
                     'goal_instr': goal_instr,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'reward': float(reward)}
        if success:
            successes.append(log_entry)
        else:
            failures.append(log_entry)

        # overall results
        results['all'] = cls.get_metrics(successes, failures)

        print("-------------")
        print("SR: %d/%d = %.3f" % (results['all']['success']['num_successes'],
                                    results['all']['success']['num_evals'],
                                    results['all']['success']['success_rate']))
        print("GC: %d/%d = %.3f" % (results['all']['goal_condition_success']['completed_goal_conditions'],
                                    results['all']['goal_condition_success']['total_goal_conditions'],
                                    results['all']['goal_condition_success']['goal_condition_success_rate']))
        print("PLW SR: %.3f" % (results['all']['path_length_weighted_success_rate']))
        print("PLW GC: %.3f" % (results['all']['path_length_weighted_goal_condition_success_rate']))
        print("-------------")

        # task type specific results
        task_types = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                      'pick_cool_then_place_in_recep', 'pick_two_obj_and_place', 'look_at_obj_in_light',
                      'pick_and_place_with_movable_recep']
        for task_type in task_types:
            task_successes = [s for s in (list(successes)) if s['type'] == task_type]
            task_failures = [f for f in (list(failures)) if f['type'] == task_type]
            if len(task_successes) > 0 or len(task_failures) > 0:
                results[task_type] = cls.get_metrics(task_successes, task_failures)
            else:
                results[task_type] = {}

        lock.release()

    @classmethod
    def get_metrics(cls, successes, failures):
        '''
        compute overall succcess and goal_condition success rates along with path-weighted metrics
        '''
        # stats
        num_successes, num_failures = len(successes), len(failures)
        num_evals = len(successes) + len(failures)
        total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                                sum([entry['path_len_weight'] for entry in failures])
        completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                   sum([entry['completed_goal_conditions'] for entry in failures])
        total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                               sum([entry['total_goal_conditions'] for entry in failures])

        # metrics
        sr = float(num_successes) / num_evals
        pc = completed_goal_conditions / float(total_goal_conditions)
        plw_sr = (float(sum([entry['path_len_weighted_success_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_success_spl'] for entry in failures])) /
                  total_path_len_weight)
        plw_pc = (float(sum([entry['path_len_weighted_goal_condition_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_goal_condition_spl'] for entry in failures])) /
                  total_path_len_weight)

        # result table
        res = dict()
        res['success'] = {'num_successes': num_successes,
                          'num_evals': num_evals,
                          'success_rate': sr}
        res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                        'total_goal_conditions': total_goal_conditions,
                                        'goal_condition_success_rate': pc}
        res['path_length_weighted_success_rate'] = plw_sr
        res['path_length_weighted_goal_condition_success_rate'] = plw_pc

        return res

    def create_stats(self):
            '''
            storage for success, failure, and results info
            '''
            self.successes, self.failures = self.manager.list(), self.manager.list()
            self.results = self.manager.dict()

    def save_results(self):
        results = {'successes': list(self.successes),
                   'failures': list(self.failures),
                   'results': dict(self.results)}

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'task_results_' + self.args.eval_split + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)

