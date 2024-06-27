from __future__ import annotations
from typing import Literal, TypedDict


class T_PSICVAnnDatabase(TypedDict):
    """Schema for the `cv_annotation.json` portion of the PSI dataset.

    :var str video_name: Name of the video.
    :var frames: Dictionary of [frame_id, annotations] pairs.
    :vartype frames: dict[str, T_FrameAnnotation | None]
    """

    video_name: str
    frames: dict[str, T_FrameAnnotation | None]


# Path: root.frame_id.cv_annotation.<type>_track_<tid>
class T_TrafficTrack(TypedDict):
    """Object containing traffic object track data.

    Path: `root.frame_<fid>.cv_annotation.<object_type>_track_<tid>`

    :var str object_type: Type of traffic object.
    :var str track_id: Track ID.
    :var str groupId: No idea.
    :var list[float] bbox: Bounding box coordinates, length 4.
    :var list[int] observed_frames: List of frames when the tracked object was observed.
    """

    object_type: str
    track_id: str
    groupId: str
    bbox: list[float]  # length of 4
    observed_frames: list[int]  # arbitrary length


# Path: root.frame_<fid>.cv_annotation.pedestrian_track_<tid>
class T_PedestrianTrack(T_TrafficTrack):
    """Object containing pedestrian type track data.

    Path: `root.frame_<fid>.cv_annotation.pedestrian_track_<tid>`

    :var T_PedestrianSkeleton skeleton: Pose data.
    """

    skeleton: T_PedestrianSkeleton


# We have to do it this way because of the brackets in the key name.
# Path: root.frame_id
T_FrameAnnotation = TypedDict(
    "T_FrameAnnotation",
    {
        "cv_annotation": dict[str, T_TrafficTrack | T_PedestrianTrack],
        "speed(km/hr)": float,
        "gps": tuple[str, str],
        "time": str,
    },
)
"""Object containing frame annotations.

Path: `root.frame_<fid>`

:var dict[str, T_TrafficTrack] cv_annotation: Computer Vision annotations.
:var float speed(km/hr): Speed in Km/h.
:var tuple[str, str] gps: GPS Coordinates.
:var str time: Time of day.
"""


# Path: root.frame_<fid>.cv_annotation.pedestrian_track_<tid>.skeleton
class T_PedestrianSkeleton(TypedDict):
    """Object containing pose information.

    Path: `root.frame_<fid>.cv_annotation.pedestrian_track_<tid>.skeleton`

    :var T_KeyPoint nose: Nose
    :var T_KeyPoint left_eye: Left Eye
    :var T_KeyPoint right_eye: Right Eye
    :var T_KeyPoint left_ear: Left Ear
    :var T_KeyPoint right_ear: Right Ear
    :var T_KeyPoint left_shoulder: Left Shoulder
    :var T_KeyPoint right_shoulder: Right Shoulder
    :var T_KeyPoint left_elbow: Left Elbow
    :var T_KeyPoint right_elbow: Right Elbow
    :var T_KeyPoint left_wrist: Left Wrist
    :var T_KeyPoint right_wrist: Right Wrist
    :var T_KeyPoint left_hip: Left Hip
    :var T_KeyPoint right_hip: Right Hip
    :var T_KeyPoint left_knee: Left Knee
    :var T_KeyPoint right_knee: Right Knee
    :var T_KeyPoint left_ankle: Left Ankle
    :var T_KeyPoint right_ankle: Right Ankle
    """

    nose: T_KeyPoint
    left_eye: T_KeyPoint
    right_eye: T_KeyPoint
    left_ear: T_KeyPoint
    right_ear: T_KeyPoint
    left_shoulder: T_KeyPoint
    right_shoulder: T_KeyPoint
    left_elbow: T_KeyPoint
    right_elbow: T_KeyPoint
    left_wrist: T_KeyPoint
    right_wrist: T_KeyPoint
    left_hip: T_KeyPoint
    right_hip: T_KeyPoint
    left_knee: T_KeyPoint
    right_knee: T_KeyPoint
    left_ankle: T_KeyPoint
    right_ankle: T_KeyPoint


# Path: root.frame_<fid>.cv_annotation.pedestrian_track_<tid>.skeleton.<part>
class T_KeyPoint(TypedDict):
    """Object containing pose keypoint information

    Path: `root.frame_<fid>.cv_annotation.pedestrian_track_<tid>.skeleton.<part>`

    :var keyframe: Flag indicating if it is a keyframe? (I think).
    :vartype keyframe: int | None
    :var str points: Screen coordinates in format `float,float`. Split with `str.split`.
    """

    keyframe: int | None  # Underlying type is a str.
    points: str  # format `float,float`


class T_PSIPedIntentCognitiveKeyframeDB(TypedDict):
    """Schema for the `pedestrian_intent.json` portion of the PSI dataset.

    :var str video_name: Name of the video.
    :var pedestrians: Dictionary of [track_<tid>, track annot dict] pairs.
    :vartype pedestrians: dict[str, T_CognitiveTrack]
    """

    video_name: str
    pedestrians: dict[str, T_CognitiveTrack]


class T_CognitiveTrack(TypedDict):
    """Cognitive annotation data for a pedestrian track.

    Path: `root.pedestrians.track_<tid>`

    :var list[int] observed_frames: List of frames when the tracked ped was observed.
    :var cv_annotations: Bounding boxes (and pose data after processing)
    :vartype cv_annotations: T_TrackBboxes | T_TrackAnnotations
    :var cognitive_annotations: Cognitive reasoning annotations.
    :vartype cognitive_annotations: dict[str, T_CognitiveAnnotation]
    """

    observed_frames: list[int]
    cv_annotations: T_TrackBboxes | T_TrackAnnotations
    cognitive_annotations: dict[str, T_CognitiveAnnotation]


class T_TrackBboxes(TypedDict):
    """Container of only bounding box data.

    Path: `root.pedestrians.track_<tid>.cv_annotations`

    :var list[list[float]] bboxes: List of bounding box coordinates, inner length 4.
    """

    bboxes: list[list[float]]  # arbirtary outer dim, inner dim 4.


class T_TrackAnnotations(T_TrackBboxes):
    """Container of bounding box and pose data.

    Path: `root.pedestrians.track_<tid>.cv_annotations`

    :var list[T_PedestrianSkeleton] skeleton: List of pose data.
    """

    skeletons: list[T_PedestrianSkeleton]


class T_CognitiveAnnotation(TypedDict):
    """Cognitive annotation container.

    Path: `root.pedestrians.track_<tid>.cognitive_annotations.<annot_id>`

    :var intent: List of pedestrian's intention for each frame.
    :vartype intent: List[Literal["", "not_cross", "not_sure", "cross"]]
    :var list[str] description: List of natural language descriptions of pedestrian in
        each frame.
    :var list[int] key_frame: List of indicators whether a frame is a keyframe.
    :var promts: More reasoning information about the description.
    :vartype promts: list[Literal[""] | T_Prompts]
    :var selectedTextBbox: Bounding boxes for prompts reasoning locality. The bounding
        box coordinates length is 4.
    :vartype selectedTextBbox: list[Literal[""] | dict[str, list[float]]]
    """

    intent: list[Literal["", "not_cross", "not_sure", "cross"]]
    description: list[str]
    key_frame: list[int]
    promts: list[Literal[""] | T_Prompts]
    selectedTextBbox: list[
        Literal[""] | dict[str, list[float]]
    ]  # the list[float] part is of length 4


class T_Prompts(TypedDict):
    """Additional reasoning annotations for the frame.

    Path: `root.pedestrians.track_<tid>.cognitive_annotations.<annot_id>.promts`

    :var str pedestrian: Reasonings for the pedestrian.
    :var str goalRelated: Reasonings related to the prediction task.
    :var str roadUsersRelated: Reasonings related to other road users.
    :var str roadFactors: Reasonings related to properties of the road.
    :var str norms: Reasonings related to road user norms.
    """

    pedestrian: str
    goalRelated: str
    roadUsersRelated: str
    roadFactors: str
    norms: str
