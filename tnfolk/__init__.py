"""Tamil Nadu Folk Song Dataset Pipeline.

Local-first, resumable pipeline that scrapes Tamil folk songs from YouTube,
runs each candidate through a fixed chain of quality/relevance gates, and only
transcribes + stores candidates that reach ACCEPT.

Pipeline order (Design Doc Section 4), do not reorder:
    Folk Category -> Query Generator -> YouTube Search -> Metadata Relevance Gate
    -> Download Audio -> Integrity Gate -> Audio Quality Gate
    -> Music/Speech/Mixed Classifier -> VAD -> Folk-Relevance Gate
    -> Duplicate Detection -> Weighted Final Scoring -> ACCEPT/REVIEW/REJECT
    -> [ACCEPT only] Full Transcription -> Local Storage
"""

__version__ = "1.0.0"
