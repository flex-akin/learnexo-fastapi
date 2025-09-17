# main.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from src.notebook_code import LearningRecommender, make_openai_client

app = FastAPI(
    title="LearNEXO FastAPI",
    version="1.0.0",
    description="Mastery-based recommendations with optional LLM feedback",
)


# ---------- Pydantic models ----------
class TopicIn(BaseModel):
    topic_id: str
    name: str
    tags: List[str] = Field(default_factory=list)
    prerequisite: Optional[str] = None


class PredictRequest(BaseModel):
    student: str
    scores: Dict[str, int]  # keys = topic_id, values = 0..100
    topics: Optional[List[TopicIn]] = None
    mastery_threshold: Optional[int] = 70
    enrich_with_llm: Optional[bool] = True

    @validator("scores")
    def _scores_nonempty(cls, v):
        if not v:
            raise ValueError("scores must not be empty")
        return v


# ---------- Default topics (English) ----------
DEFAULT_TOPICS: List[Dict[str, Any]] = [
    {
        "topic_id": "read",
        "name": "Reading",
        "tags": ["english", "read"],
        "prerequisite": None,
    },
    {
        "topic_id": "Writing",
        "name": "Writing",
        "tags": ["english", "write"],
        "prerequisite": None,
    },
    {
        "topic_id": "list-speak",
        "name": "Listening & Speaking",
        "tags": ["english", "list-speak"],
        "prerequisite": None,
    },
]


# ---------- Engine boot ----------
openai_client = make_openai_client()
recommender = LearningRecommender(
    topics=DEFAULT_TOPICS,
    mastery_threshold=70,
    openai_client=openai_client,
    openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
)


# ---------- Endpoint ----------
@app.post("/predict")
def predict(req: PredictRequest):
    """
    POST /predict

    Request scores are keyed by topic_id. We map those IDs to topic names
    using the provided topics (or defaults), then call a single method on
    the LearningRecommender to produce recommendations (optionally with LLM
    feedback if configured and requested).
    """
    try:
        topics_list = [t.dict() for t in req.topics] if req.topics else DEFAULT_TOPICS
        topic_id_to_name = {t["topic_id"]: t["name"] for t in topics_list}

        # Map topic_id->score to topic_name->score
        scores_by_name: Dict[str, int] = {}
        unknown_ids: List[str] = []
        for tid, score in req.scores.items():
            name = topic_id_to_name.get(tid)
            if name is None:
                unknown_ids.append(tid)
            else:
                scores_by_name[name] = int(score)

        if not scores_by_name:
            raise HTTPException(
                status_code=400,
                detail="No valid topic IDs found in scores. Please check your request.",
            )

        # Create a fresh engine if request overrides topics or threshold
        engine = recommender
        if req.topics or (
            req.mastery_threshold is not None
            and req.mastery_threshold != recommender.mastery_threshold
        ):
            engine = LearningRecommender(
                topics=topics_list,
                mastery_threshold=req.mastery_threshold or 70,
                openai_client=openai_client,
                openai_model=recommender.openai_model,
            )

        recs = engine.recommend(
            student=req.student,
            scores_by_name=scores_by_name,
            enrich_with_llm=bool(req.enrich_with_llm),
        )

        response: Dict[str, Any] = {"ok": True, "recommendations": recs}
        if unknown_ids:
            response["warnings"] = {"unknown_topic_ids": unknown_ids}
        return response

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500, detail=f"Error while running prediction: {e}\n{tb}"
        )


if __name__ == "__main__":
    # Run: uvicorn main:app --reload
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
