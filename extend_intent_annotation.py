from __future__ import annotations
from collections.abc import Sequence
import copy
import json
import os
import sys

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


def amend_keyframe_annotations(
    cog_track: T_CognitiveTrack, last_key_frames: list[int], observed_frames: list[int]
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

    poses = cog_track["cv_annotations"].get("skeletons", None)
    if poses is None:
        cv_annot = cog_track["cv_annotations"]
        raise RuntimeError(f"Pose data is empty. {cv_annot}")
    else:
        del poses[last_intent_estimate_idx + 1 :]

    for _, cog_ann in cog_track["cognitive_annotations"].items():
        del cog_ann["intent"][last_intent_estimate_idx + 1 :]
        del cog_ann["key_frame"][last_intent_estimate_idx + 1 :]
        del cog_ann["description"][last_intent_estimate_idx + 1 :]


def main():
    print("Extend Intent Annotations of PSI 2.0 Dataset.")

    root_path = sys.argv[1]

    key_frame_anotation_path = os.path.join(
        root_path, "PSI2.0_TrainVal/annotations/cognitive_annotation_key_frame"
    )
    extended_annotation_path = os.path.join(
        root_path, "PSI2.0_TrainVal/annotations/cognitive_annotation_extended"
    )
    cv_annotation_path = os.path.join(
        root_path, "PSI2.0_TrainVal/annotations/cv_annotation"
    )

    if not os.path.exists(extended_annotation_path):
        os.makedirs(extended_annotation_path)

    video_list = sorted(os.listdir(key_frame_anotation_path))

    for vname in tqdm(video_list, desc="Video", position=0, leave=True):
        # 1. load key-frame annotations
        key_intent_ann_file = os.path.join(
            key_frame_anotation_path, vname, "pedestrian_intent.json"
        )
        with open(key_intent_ann_file, "r") as f:
            key_intent_ann: T_PSIPedIntentCognitiveKeyframeDB = json.load(f)

        # 1.2 load cv_annotations
        cv_ann_file = os.path.join(cv_annotation_path, vname, "cv_annotation.json")
        with open(cv_ann_file, "r", encoding="utf-8") as f:
            cv_ann: T_PSICVAnnDatabase = json.load(f)

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
            # `extended_intent_ann.pedestrians.track_<tid>.cv_annotations.skeletons`
            # TODO: Verify that it works. (It doesn't)
            # TODO: Interpolate between pose keyframes.
            observed_frames = ped_track["observed_frames"]
            for fid in tqdm(
                observed_frames, desc="Extracting Poses", position=2, leave=False
            ):
                if (
                    pose_data := cv_ann["frames"][f"frame_{fid}"]["cv_annotation"][
                        f"pedestrian_{ped_k}"
                    ].get("skeleton", None)
                ) is not None:
                    if (
                        skeletons := ped_track["cv_annotations"].get("skeletons", None)
                    ) is None:
                        ped_track["cv_annotations"]["skeletons"] = [pose_data]
                    else:
                        skeletons.append(pose_data)

            # stuff for asserts
            # following line should error if empty
            skeletons = ped_track["cv_annotations"]["skeletons"]
            bboxes = ped_track["cv_annotations"]["bboxes"]
            assert len(skeletons) == len(
                bboxes
            ), f"len(skeletons): {len(skeletons)}, len(bboxes): {len(bboxes)}"
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
                amend_keyframe_annotations(cog_track, last_key_frames, observed_frames)

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

        print(
            vname,
            ": Original observed frames: {} --> valid intent estimation frames: {}".format(
                len(key_intent_ann["pedestrians"][ped_k]["observed_frames"]),
                len(extended_intent_ann["pedestrians"][ped_k]["observed_frames"]),
            ),
        )


if __name__ == "__main__":
    main()
