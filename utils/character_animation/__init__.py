"""Character-consistent, frame-by-frame animation pipeline.

The package is additive: it does not modify the classic screenplay pipeline,
so existing screenplays and outputs remain valid.
"""

from utils.character_animation.character_board import CharacterBoardBuilder
from utils.character_animation.character_designer import CharacterDesigner
from utils.character_animation.character_inputs import CharacterInputLibrary
from utils.character_animation.frame_animator import FrameAnimator
from utils.character_animation.frame_video_builder import FrameVideoBuilder
from utils.character_animation.processor import CharacterAnimationProcessor
from utils.character_animation.quality_evaluator import (
    EvaluationReport,
    EvaluationThresholds,
    FrameAnalysis,
    SceneEvaluation,
    VideoQualityEvaluator,
)
from utils.character_animation.qwen_edit_engine import QwenImageEditEngine
from utils.character_animation.screenplay import (
    CharacterScreenplay,
    CharacterSpec,
    SceneSpec,
    scale_scene_durations,
    slugify,
)

__all__ = [
    "CharacterAnimationProcessor",
    "CharacterBoardBuilder",
    "CharacterDesigner",
    "CharacterInputLibrary",
    "CharacterScreenplay",
    "CharacterSpec",
    "EvaluationReport",
    "EvaluationThresholds",
    "FrameAnalysis",
    "FrameAnimator",
    "FrameVideoBuilder",
    "QwenImageEditEngine",
    "SceneEvaluation",
    "SceneSpec",
    "VideoQualityEvaluator",
    "scale_scene_durations",
    "slugify",
]
