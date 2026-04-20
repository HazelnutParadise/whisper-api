from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable error message.")


class WordTimestamp(BaseModel):
    word: str = Field(description="Recognized word token.")
    start: float | None = Field(
        default=None,
        description="Word start time in seconds.",
    )
    end: float | None = Field(
        default=None,
        description="Word end time in seconds.",
    )
    speaker: str | None = Field(
        default=None,
        description="Speaker label when diarization is enabled.",
    )


class SegmentTimestamp(BaseModel):
    id: int | None = Field(default=None, description="Segment index.")
    start: float | None = Field(
        default=None,
        description="Segment start time in seconds.",
    )
    end: float | None = Field(
        default=None,
        description="Segment end time in seconds.",
    )
    text: str = Field(description="Transcript text for this segment.")
    speaker: str | None = Field(
        default=None,
        description="Speaker label when diarization is enabled.",
    )
    words: list[WordTimestamp] | None = Field(
        default=None,
        description="Per-word timestamps when alignment is available.",
    )


class DiarizationSegment(BaseModel):
    speaker: str | None = Field(
        default=None,
        description="Speaker label such as SPEAKER_00.",
    )
    start: float | None = Field(
        default=None,
        description="Speaker segment start time in seconds.",
    )
    end: float | None = Field(
        default=None,
        description="Speaker segment end time in seconds.",
    )


class TranscriptionSimpleResponse(BaseModel):
    text: str = Field(description="Plain transcription text.")


class TranscriptionAdvancedResponse(TranscriptionSimpleResponse):
    language: str = Field(description="Detected language code.")
    segments: list[SegmentTimestamp] = Field(
        description="Aligned transcript segments with timestamps.",
    )
    diarization: list[DiarizationSegment] = Field(
        description="Speaker diarization segments.",
    )
    speakers: list[str] = Field(
        description="Unique speaker labels that appear in the audio.",
    )
