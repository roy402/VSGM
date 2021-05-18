import imageio
import os
import torch
import numpy as np
import cv2
SAVE_FOLDER_NAME = "eval_video"
font = cv2.FONT_HERSHEY_SIMPLEX
toptomLeftCornerOfText = (10, 20)
topmiddleLeftCornerOfText = (10, 50)
middleLeftCornerOfText = (10, 230)
bottomLeftCornerOfText = (10, 270)
topmiddleOfText = (600, 30)
fontScale = 0.7
r_fontColor = (255, 0, 0)
g_fontColor = (0, 255, 0)
lineType = 2


class EvalDebug():
    """docstring for EvalDebug"""
    def __init__(self):
        super(EvalDebug, self).__init__()
        self.reset_data()

    def reset_data(self):
        self.images = []
        self.depths = []
        self.list_actions = []
        self.lang_instr = []
        self.fail_reason_list = []
        self.fail_reason = ""

    def add_data(self, step, image, depth, dict_action, lang_instr, fail_reason):
        if fail_reason != "":
            # string-
            fail_reason = str(fail_reason)
            self.fail_reason += str(step) + ": " + fail_reason
            self.fail_reason += "\n"
        else:
            fail_reason = "."
        if dict_action["mask"] is None:
            dict_action["mask"] = (np.ones(depth.shape)*255).astype(np.uint8)
        else:
            dict_action["mask"] = (dict_action["mask"]*255).astype(np.uint8)
        self.images.append(image)
        self.depths.append(depth)
        self.list_actions.append(dict_action)
        self.lang_instr.append(lang_instr)
        self.fail_reason_list.append(fail_reason)


    def store_state_case(self, file_name, save_dir, goal_instr, step_instr):
        save_fail_case = os.path.join(save_dir, file_name + "_state.txt")
        with open(save_fail_case, 'w') as f:
            f.write("Save Dir: " + save_fail_case)
            f.write("\nGoal: " + goal_instr + "\nstep_instr: ")
            f.write("\nstep_instr: ".join(step_instr) + "\n")
            f.write(self.fail_reason)

    def store_current_state(self, file_name, save_dir, text):
        save_fail_case = os.path.join(save_dir, file_name + "_state.txt")
        with open(save_fail_case, 'a') as f:
            f.write("\n" + text)

    def record(self, save_dir, traj_data, goal_instr, step_instr, fail_reason, success, fps=2, eval_idx=""):
        # path
        if success:
            file_name = "S_"
        else:
            file_name = "F_"
        file_name += str(traj_data['repeat_idx']) + eval_idx
        fold_name = traj_data['task_type'] + '_' + traj_data['task_id']

        save_dir = os.path.join(save_dir, SAVE_FOLDER_NAME, fold_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.store_state_case(file_name, save_dir, goal_instr, step_instr)
        self.images_to_video(file_name, save_dir, goal_instr, fps)
        self.reset_data()

    def images_to_video(self, file_name, save_dir, goal_instr, fps):
        v_file_name = file_name + ".mp4"
        save_video_dir = os.path.join(save_dir, v_file_name)
        writer = imageio.get_writer(save_video_dir, fps=fps)
        i = 0
        # data
        for image, depth, dict_action, lang_instr, fail_reason in\
            zip(self.images, self.depths, self.list_actions, self.lang_instr, self.fail_reason_list):
            '''
            Process image
            '''
            depth = np.expand_dims(depth, axis=2)
            depth = np.tile(depth, (1, 3))
            mask = np.expand_dims(dict_action["mask"], axis=2)
            mask = np.tile(mask, (1, 3))
            cat_image = np.concatenate([image, depth, mask], axis=1)
            '''
            Process string
            '''
            # action
            if len(dict_action["action_navi_or_operation"])<1\
                or dict_action["action_navi_or_operation"][0, 0]>dict_action["action_navi_or_operation"][0, 1]:
                color = r_fontColor
            else:
                color = g_fontColor
            if len(dict_action["action_navi_or_operation"])>0:
                p = dict_action["action_navi_or_operation"].tolist()
                p = np.round(p, decimals=2)
            else:
                p = []
            str_step = "step: " + str(i)
            str_action_low = dict_action["action_low"]
            str_navi_oper = "navi: " + dict_action["action_navi_low"] +\
                ", oper: " + dict_action["action_operation_low"]
            str_p_navi_or_operation = "p navi/oper: " + str(p)
            str_global_graph_dict_ANALYZE_GRAPH = "global " + str(dict_action["global_graph_dict_ANALYZE_GRAPH"]).replace(":", "").replace(" ", "")
            str_current_state_dict_ANALYZE_GRAPH = "current " + str(dict_action["current_state_dict_ANALYZE_GRAPH"]).replace(":", "").replace(" ", "")
            str_history_changed_dict_ANALYZE_GRAPH = "history " + str(dict_action["history_changed_dict_ANALYZE_GRAPH"]).replace(":", "").replace(" ", "")
            str_priori_dict_ANALYZE_GRAPH = "priori " + str(dict_action["priori_dict_ANALYZE_GRAPH"]).replace(":", "").replace(" ", "")
            str_mask = str(dict_action["pred_class"]) + ", " + dict_action["object"]
            str_subgoal_progress = str(dict_action["subgoal_t"]) + ", " + str(dict_action["progress_t"])
            self.store_current_state(file_name, save_dir, str_step)
            self.store_current_state(file_name, save_dir, str_action_low)
            self.store_current_state(file_name, save_dir, str_mask)
            self.store_current_state(file_name, save_dir, "Fail: " + fail_reason)
            self.store_current_state(file_name, save_dir, str_global_graph_dict_ANALYZE_GRAPH)
            self.store_current_state(file_name, save_dir, str_current_state_dict_ANALYZE_GRAPH)
            self.store_current_state(file_name, save_dir, str_history_changed_dict_ANALYZE_GRAPH)
            self.store_current_state(file_name, save_dir, str_priori_dict_ANALYZE_GRAPH)
            self.store_current_state(file_name, save_dir, str_subgoal_progress)
            i += 1
            '''
            write
            '''
            self.writeText(
                cat_image, str_step, toptomLeftCornerOfText, color)
            self.writeText(
                cat_image, str_action_low, (toptomLeftCornerOfText[0]+100, toptomLeftCornerOfText[1]), color)
            '''
            # navi oper
            self.writeText(
                cat_image, str_navi_oper, topmiddleLeftCornerOfText, r_fontColor)
            # action_navi_or_operation
            self.writeText(
                cat_image, str_p_navi_or_operation, (topmiddleLeftCornerOfText[0], topmiddleLeftCornerOfText[1]+30), r_fontColor)
            '''
            # ANALYZE_GRAPH
            '''
            self.writeText(
                cat_image, str_global_graph_dict_ANALYZE_GRAPH, (topmiddleLeftCornerOfText[0], topmiddleLeftCornerOfText[1]+60), r_fontColor, fontscale=0.6)
            self.writeText(
                cat_image, str_current_state_dict_ANALYZE_GRAPH, (topmiddleLeftCornerOfText[0], topmiddleLeftCornerOfText[1]+90), r_fontColor, fontscale=0.6)
            self.writeText(
                cat_image, str_history_changed_dict_ANALYZE_GRAPH, (topmiddleLeftCornerOfText[0], topmiddleLeftCornerOfText[1]+120), r_fontColor, fontscale=0.6)
            self.writeText(
                cat_image, str_priori_dict_ANALYZE_GRAPH, (topmiddleLeftCornerOfText[0], topmiddleLeftCornerOfText[1]+150), r_fontColor, fontscale=0.6)

            # fail_reason
            self.writeText(
                cat_image, fail_reason, middleLeftCornerOfText, r_fontColor)
            # goal_instr
            # self.writeText(
            #     cat_image, goal_instr, bottomLeftCornerOfText, r_fontColor)
            self.writeText(
                cat_image, lang_instr, (bottomLeftCornerOfText[0], bottomLeftCornerOfText[1]+20), r_fontColor)
            # dict_mask
            self.writeText(
                cat_image, str_mask, topmiddleOfText, r_fontColor)
            # goal persent: subgoal_t progress_t
            self.writeText(
                cat_image, str_subgoal_progress, (topmiddleOfText[0], topmiddleOfText[1]+270), r_fontColor)
            '''
            writer.append_data(cat_image)
        writer.close()

    def writeText(self, img, string, position, color, fontscale=fontScale):
        cv2.putText(img, string,
                    position,
                    font,
                    fontscale,
                    color,
                    lineType)