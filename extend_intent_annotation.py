from __future__ import annotations
import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import copy
import json
import os
import sys
from typing import Any, cast

from tqdm.auto import tqdm

from cv_annotation_schema import *

POSE_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# TODO: (chris) Write posture data to the extended annotations.


def most_frequent(array: Sequence[str | int]):
    # return the most frequent intent estimation made by all annotators
    counter = 0
    num = array[0]

    for i in array:
        curr_frequency = array.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num


def keypoints_to_indexed(
    dataset: Literal["PSI1.0", "PSI2.0"],
    keypoints: T_PedestrianSkeleton1 | T_PedestrianSkeleton2,
) -> tuple[list[tuple[float, float]], list[bool]]:
    # 1. Create "observed" mask for pose.
    observed_pose_data: list[bool] = [True] * 17

    # 2. Create missing fields if any.
    for i, keypoint_name in enumerate(POSE_KEYPOINTS):
        if keypoints.get(keypoint_name, None) is None:
            keypoints[keypoint_name] = (0.0, 0.0)
            observed_pose_data[i] = False

        elif isinstance(keypoints[keypoint_name], dict):
            if keypoints[keypoint_name].get("points", None) is None:
                keypoints[keypoint_name]["points"] = (0.0, 0.0)
                observed_pose_data[i] = False

        # 3. If PSI2.0, convert fields to float.
        match dataset:
            case "PSI2.0":
                match (keypoints[keypoint_name]):
                    case dict():
                        try:
                            coords = keypoints[keypoint_name]["points"]
                            x_str, y_str = coords.split(",")
                            x = float(x_str)
                            y = float(y_str)
                            keypoints[keypoint_name] = (x, y)
                        except:
                            keypoints[keypoint_name] = (0.0, 0.0)
                            observed_pose_data[i] = False
                    case None:
                        keypoints[keypoint_name] = (0.0, 0.0)
                        observed_pose_data[i] = False
                    case _:
                        raise TypeError(
                            f"Type of keypoints[keypoint_name] ({type(keypoints[keypoint_name])}) is not valid"
                        )
            case _:
                pass

    assert len(keypoints) == len(
        POSE_KEYPOINTS
    ), f"Number of filled in keypoints {len(keypoints)} is less than {len(POSE_KEYPOINTS)}"

    # 4. Convert to indexed array based on `POSE_KEYPOINTS`.
    ret: list[tuple[float, float]] = []

    for i, keypoint_name in enumerate(POSE_KEYPOINTS):
        keypoint = keypoints[keypoint_name]
        if keypoint == (0.0, 0.0):
            assert not observed_pose_data[
                i
            ], f"Observed mask at index {i} must be False!"
        ret.append(keypoints[keypoint_name])

    return ret, observed_pose_data


def amend_keyframe_annotations(
    args: ScriptArguments,
    cog_track: T_CognitiveTrack,
    last_key_frames: list[int],
    observed_frames: list[int],
):
    bboxes = cog_track["cv_annotations"]["bboxes"]

    start_box = bboxes[0]
    last_key_box = bboxes[max(last_key_frames)]
    end_box = bboxes[-1]

    last_intent_estimate_idx: int = 0
    if ((last_key_box[0] + last_key_box[2]) / 2 - 640) * (
        (end_box[0] + end_box[2]) / 2 - 640
    ) >= 0:  # ? where did these constants come from

        # Situation 1: By the last key-frame annotation, the target pedestrian already
        # crossed the middle line of the ego-view, and is on the same side as the
        # final position.
        # In such case, we use the last annotated key-frame as the end of "intent
        # estimation" task
        last_intent_estimate_idx = max(last_key_frames)

    else:  # < 0
        # Situation 2: By the last key-frame annotation, the target pedestrian is at a
        # different position compared to the final observed position.
        # In such case, we use the moment when the target pedestrian crossed the middle
        # line of the ego-view as the last frame of "intent estimation" task
        for cur_frame_k in range(max(last_key_frames), len(observed_frames)):
            # pedestrian could change positions several times, e.g., vehicle is
            # turning. Thus starts from the last key-frame
            current_box = cog_track["cv_annotations"]["bboxes"][cur_frame_k]
            if ((current_box[0] + current_box[2]) / 2 - 640) * (
                (end_box[0] + end_box[2]) / 2 - 640
            ) >= 0:
                # once the pedestrian crossed the middle line of ego-view, to the same
                # side as the last frame, use this moment as the last intent estimation
                # task frame
                last_intent_estimate_idx = cur_frame_k
                break
            else:
                continue

    # Cut redundant intent extended annotations that not usable for "intent estimation" task
    del cog_track["observed_frames"][last_intent_estimate_idx + 1 :]
    del bboxes[last_intent_estimate_idx + 1 :]

    poses = cog_track["cv_annotations"].get("skeleton", None)
    if poses is None:
        cv_annot = cog_track["cv_annotations"]
        if args.allow_empty_poses:
            num_frames = len(bboxes)
            cog_track["cv_annotations"]["skeleton"] = [[0.0, 0.0] * num_frames]
        else:
            raise RuntimeError(f"Pose data is empty. {cv_annot.keys()}")
    else:
        del poses[last_intent_estimate_idx + 1 :]

    for _, cog_ann in cog_track["cognitive_annotations"].items():
        del cog_ann["intent"][last_intent_estimate_idx + 1 :]
        del cog_ann["key_frame"][last_intent_estimate_idx + 1 :]
        del cog_ann["description"][last_intent_estimate_idx + 1 :]


def main(args: ScriptArguments):
    print("Extend Intent Annotations of PSI 2.0 Dataset.")

    root_path: str = args.root_dir
    dataset: str = args.dataset

    dataset_path: str = ""
    match dataset:
        case "PSI2.0":
            dataset_path = "PSI2.0_TrainVal"
        case "PSI1.0":
            dataset_path = "PSI1.0"
        case _:
            raise NotImplementedError

    key_frame_annotation_path = os.path.join(
        root_path, dataset_path, "annotations/cognitive_annotation_key_frame"
    )
    extended_annotation_path = os.path.join(
        root_path, dataset_path, "annotations/cognitive_annotation_extended"
    )
    cv_annotation_path = os.path.join(
        root_path, dataset_path, "annotations/cv_annotation"
    )

    if not os.path.exists(extended_annotation_path):
        os.makedirs(extended_annotation_path)

    video_list = sorted(os.listdir(key_frame_annotation_path))

    mbar = tqdm(video_list, desc="Video", position=0, leave=True)
    for vname in mbar:
        # 1. load key-frame annotations
        if os.path.exists(
            kf_ann_path := os.path.join(
                key_frame_annotation_path, vname, "pedestrian_intent.json"
            )
        ):
            with open(kf_ann_path, "r", encoding="utf-8") as f:
                key_intent_ann: T_PSIPedIntentCognitiveKeyframeDB = json.load(f)
        else:
            mbar.write(f"Keyframe annotations for {vname} not found.")
            continue

        # 1.2 load cv_annotations
        if os.path.exists(
            cv_ann_file := os.path.join(cv_annotation_path, vname, "cv_annotation.json")
        ):
            with open(cv_ann_file, "r", encoding="utf-8") as f:
                cv_ann: T_PSICVAnnDatabase = json.load(f)
        else:
            mbar.write(f"CV annotations for {vname} not found.")
            continue

        # 2. extend annotations (intent & description) - intent to the future frames,
        # description to the prior frames
        extended_intent_ann = copy.deepcopy(key_intent_ann)

        ped_k: str = ""

        for ped_k, ped_track in tqdm(
            key_intent_ann["pedestrians"].items(),
            desc="Pedstrian",
            position=1,
            leave=False,
        ):
            # Retrieve the pose data from
            # `cv_ann.frames.frame_<fid>.cv_annotation.pedestrian_track_<tid>.skeleton`
            # and insert all frames into
            # `extended_intent_ann.pedestrians.track_<tid>.cv_annotations.skeleton`
            # TODO: Interpolate between pose keyframes.
            observed_frames = ped_track["observed_frames"]
            pose_data_key = "skeleton" if dataset == "PSI2.0" else "joints"
            for fid in tqdm(
                observed_frames, desc="Extracting Poses", position=2, leave=False
            ):  # for each frame in observed frames
                if (
                    pose_data := cv_ann["frames"][f"frame_{fid}"]["cv_annotation"][
                        f"pedestrian_{ped_k}"
                    ].get(pose_data_key, None)
                ) is not None:  # if pose data key does exist in anns
                    if (
                        skeletons := ped_track["cv_annotations"].get(
                            pose_data_key, None
                        )
                    ) is None:  # if pose data is empty
                        converted_pose_data, observed_pose_data = keypoints_to_indexed(
                            dataset, pose_data
                        )
                        ped_track["cv_annotations"][pose_data_key] = [
                            converted_pose_data
                        ]
                        if (
                            pose_mask := ped_track["cv_annotations"].get(
                                "observed_skeleton", None
                            )
                        ) is not None:  # if there is an array of masks
                            pose_mask.append(observed_pose_data)
                        else:
                            ped_track["cv_annotations"]["observed_skeleton"] = [
                                observed_pose_data
                            ]
                    else:  # otherwise, append to pose data arrays
                        converted_pose_data, observed_pose_data = keypoints_to_indexed(
                            dataset, pose_data
                        )
                        skeletons.append(converted_pose_data)
                        if (
                            pose_mask := ped_track["cv_annotations"].get(
                                "observed_skeleton", None
                            )
                        ) is not None:  # if there already is an array of masks
                            pose_mask.append(observed_pose_data)
                        else:
                            ped_track["cv_annotations"]["observed_skeleton"] = [
                                observed_pose_data
                            ]
                else:  # if it doesn't exist
                    if (
                        bbox_data := cv_ann["frames"][f"frame_{fid}"]["cv_annotation"][
                            f"pedestrian_{ped_k}"
                        ].get(
                            "bbox", None
                        )  # if bboxes also empty
                    ) is not None:
                        mbar.write(f"{fid} is observed but has no bbox nor pose data")
                    else:  # otherwise
                        if args.allow_empty_poses:  # write 0s to pose and pose mask
                            converted_pose_data = [(0.0, 0.0) * 17]
                            ped_track["cv_annotations"][pose_data_key] = [
                                converted_pose_data
                            ]
                            observed_pose_data: list[bool] = [False] * 17
                            ped_track["cv_annotations"]["observed_skeleton"] = [
                                observed_pose_data
                            ]
                        else:
                            mbar.write(f"{fid} is observed but has no pose data.")

            # stuff for asserts
            # following line should error if empty
            skeletons = ped_track["cv_annotations"][pose_data_key]
            pose_mask = ped_track["cv_annotations"]["observed_skeleton"]
            bboxes = ped_track["cv_annotations"]["bboxes"]
            assert len(skeletons) == len(
                observed_frames
            ), f"len(skeletons): {len(skeletons)}, len(observed_frames): {len(observed_frames)}"
            ped_track["cv_annotations"]["skeleton"] = skeletons
            for ann_k, cog_ann in tqdm(
                ped_track["cognitive_annotations"].items(),
                desc="Extracting cognitive annotations",
                position=2,
                leave=False,
            ):
                intent_list = cog_ann["intent"]
                key_frame_list = cog_ann["key_frame"]
                description_list = cog_ann["description"]
                assert len(intent_list) == len(key_frame_list) == len(description_list)

                # 0.5 # at the beginning if no labels, use "not_sure"
                pivot_int = "not_sure"

                for frame_k in tqdm(
                    range(len(observed_frames)),
                    desc="Extending intent keyframes",
                    position=3,
                    leave=False,
                ):
                    if intent_list[frame_k] == "":
                        extended_intent_ann["pedestrians"][ped_k][
                            "cognitive_annotations"
                        ][ann_k]["intent"][frame_k] = pivot_int
                    else:
                        pivot_int = intent_list[frame_k]

                pivot_des = description_list[-1]
                for frame_k in tqdm(
                    range(len(observed_frames) - 1, -1, -1),
                    desc="Extending keyframes (backwards)",
                    position=3,
                    leave=False,
                ):
                    if description_list[frame_k] == "":
                        extended_intent_ann["pedestrians"][ped_k][
                            "cognitive_annotations"
                        ][ann_k]["description"][frame_k] = pivot_des
                    else:
                        pivot_des = description_list[frame_k]
                    # Note: after this operation, some frames at the end of the
                    # observed frame list do not have descriptions,
                    # ['description']== ""

        # 3. Ignore 'Already-crossed' frames
        for ped_k, cog_track in tqdm(
            extended_intent_ann["pedestrians"].items(),
            desc="Ignore 'already-crossed'",
            position=1,
            leave=False,
        ):
            observed_frames = cog_track["observed_frames"]
            last_intents: list[Literal["", "not_cross", "not_sure", "cross"]] = []
            last_key_frames: list[int] = []
            for ann_k, cog_ann in tqdm(
                cog_track["cognitive_annotations"].items(),
                desc="Cognitive annotations",
                position=2,
                leave=False,
            ):
                intent_list = cog_ann["intent"]
                key_frame_list = cog_ann["key_frame"]
                last_intents.append(intent_list[-1])
                for j in tqdm(
                    range(len(observed_frames) - 1, -1, -1),
                    desc="Frames",
                    position=3,
                    leave=False,
                ):
                    if key_frame_list[j] != 0:
                        last_key_frames.append(j)
                        # NOTE: Here 'j' is not the frame number, it's the idx/
                        # position of the frame in the 'observed_frame' list

                        break
                    else:
                        continue

            # only apply to the 'cross' cases
            if most_frequent(last_intents) == "cross":
                amend_keyframe_annotations(
                    args, cog_track, last_key_frames, observed_frames
                )

        # 4. output extended annotations
        output_dir = os.path.join(extended_annotation_path, vname)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # write json to file
        extended_intent_ann_file = os.path.join(
            extended_annotation_path, vname, "pedestrian_intent.json"
        )
        with open(extended_intent_ann_file, "w") as file:
            # Is this necessary? why not just use `json.dump(extended_intent_ann, f)`
            json_string = json.dumps(
                extended_intent_ann,
                default=lambda o: o.__dict__,
                sort_keys=False,
                indent=4,
            )
            _ = file.write(json_string)

        mbar.write(
            "{}: Original observed frames: {} --> valid intent estimation frames: {}".format(
                vname,
                len(key_intent_ann["pedestrians"][ped_k]["observed_frames"]),
                len(extended_intent_ann["pedestrians"][ped_k]["observed_frames"]),
            ),
        )


@dataclass
class ScriptArguments:
    root_dir: str
    dataset: Literal["PSI1.0", "PSI2.0"] = "PSI2.0"
    allow_empty_poses: bool = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    _ = parser.add_argument("root_dir", type=str)

    _ = parser.add_argument("--dataset", default="PSI2.0", choices=["PSI2.0", "PSI1.0"])

    _ = parser.add_argument("--allow_empty_poses", action="store_true")

    args = cast(ScriptArguments, parser.parse_args())  # type: ignore[reportInvalidCast]

    main(args)
