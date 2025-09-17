# notebook_code.py
from __future__ import annotations

import os
import re
import random
from typing import Dict, List, Optional, Any, Tuple, Set

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def make_openai_client():
    """
    Returns an OpenAI client if OPENAI_API_KEY is set and the package is installed.
    Otherwise returns None and prints a warning.
    """
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not set; LLM feedback disabled.")
            return None
        return OpenAI(api_key=api_key)
    except Exception:
        print("Warning: `openai` package not available; LLM feedback disabled.")
        return None


class LearningRecommender:
    """
    Mastery-based recommender.

    - Finds topics below the mastery threshold.
    - If a weak topic has a prerequisite, recommend that prerequisite first.
    - Otherwise, try siblings by shared *specific* tags (excluding generic tags like 'english').
    - If none found, fall back to topics sharing any *generic* tag with the weak topic.
    - Optionally enrich each recommendation with a 1-sentence pep talk via OpenAI.
    """

    STYLE_BANK = [
        "Keep it upbeat and energetic.",
        "Use a calm, reassuring tone.",
        "Aim for playful and friendly.",
        "Make it bold and confident.",
        "Be warm and supportive.",
        "Make it concise but inspiring.",
    ]

    STARTERS = [
        "You've got this—",
        "Keep going—",
        "Don’t stop now—",
        "Take the next small step—",
        "Power through—",
        "One step at a time—",
    ]

    STOPWORDS = {
        "the", "a", "an", "and", "to", "of", "in", "for", "with",
        "on", "at", "by", "it", "is", "you", "your"
    }

    def __init__(
        self,
        topics: List[Dict[str, Any]],
        mastery_threshold: int = 70,
        openai_client=None,
        openai_model: str = "gpt-4o-mini",
    ):
        self.topics = topics
        self.mastery_threshold = mastery_threshold
        self.openai = openai_client
        self.openai_model = openai_model
        self._recent_feedback: List[str] = []

        self.topic_by_id: Dict[str, Dict[str, Any]] = {t["topic_id"]: t for t in topics}
        self.id_by_name: Dict[str, str] = {t["name"]: t["topic_id"] for t in topics}

        # Detect "generic" tags (tags that appear in a majority of topics, often 'english')
        self.generic_tags = self._infer_generic_tags(topics)

    def recommend(
        self,
        student: str,
        scores_by_name: Dict[str, int],
        enrich_with_llm: bool = True,
    ) -> List[Dict[str, str]]:
        df = self._build_df(student, scores_by_name)
        core = self._core_recommendations(student, df)
        if enrich_with_llm and self.openai:
            return self._add_llm_feedback(core)
        return core

    # -------------------- internals --------------------

    def _build_df(self, student: str, scores_by_name: Dict[str, int]) -> pd.DataFrame:
        rows = []
        for name, score in scores_by_name.items():
            topic_id = self.id_by_name.get(name)
            if topic_id is None:
                # Skip unknown names (resilience)
                continue
            rows.append(
                {
                    "student": student,
                    "topic_id": topic_id,
                    "topic_name": name,
                    "score": int(score),
                }
            )
        return pd.DataFrame(rows)

    def _infer_generic_tags(self, topics: List[Dict[str, Any]]) -> Set[str]:
        freq: Dict[str, int] = {}
        for t in topics:
            for tag in t.get("tags", []):
                freq[tag] = freq.get(tag, 0) + 1
        if not topics:
            return set()
        # Tag is generic if it appears in >= 50% of topics
        threshold = max(1, len(topics) // 2)
        return {tag for tag, c in freq.items() if c >= threshold}

    def _core_recommendations(self, student: str, df: pd.DataFrame) -> List[Dict[str, str]]:
        recs: List[Dict[str, str]] = []

        if df.empty:
            return recs

        weak = df[(df["student"] == student) & (df["score"] < self.mastery_threshold)]
        for _, row in weak.iterrows():
            topic_id = row["topic_id"]
            topic = self.topic_by_id.get(topic_id)
            if not topic:
                continue

            prereq_id = topic.get("prerequisite")
            if prereq_id:
                prereq = self.topic_by_id.get(prereq_id)
                if prereq:
                    recs.append(
                        {
                            "recommend_for": topic["name"],
                            "recommended_topic": prereq["name"],
                        }
                    )
                    continue  # Prefer prereq if present

            # No prereq: try siblings by *specific* tags (exclude generic tags)
            t_tags = set(topic.get("tags", []))
            specific_tags = [tg for tg in t_tags if tg not in self.generic_tags]

            matched_any = False
            if specific_tags:
                for other in self.topics:
                    if other["topic_id"] == topic_id:
                        continue
                    o_tags = set(other.get("tags", []))
                    if any(tg in o_tags for tg in specific_tags):
                        recs.append(
                            {
                                "recommend_for": topic["name"],
                                "recommended_topic": other["name"],
                            }
                        )
                        matched_any = True

            # Still nothing? Fall back to recommending others that share any generic tag
            if not matched_any and self.generic_tags:
                for other in self.topics:
                    if other["topic_id"] == topic_id:
                        continue
                    if set(other.get("tags", [])) & self.generic_tags:
                        recs.append(
                            {
                                "recommend_for": topic["name"],
                                "recommended_topic": other["name"],
                            }
                        )

        # Deduplicate
        seen: Set[Tuple[str, str]] = set()
        out: List[Dict[str, str]] = []
        for r in recs:
            key = (r["recommend_for"], r["recommended_topic"])
            if key not in seen:
                out.append(r)
                seen.add(key)
        return out

    def _normalize_text(self, t: str) -> List[str]:
        return re.sub(r"[^a-z ]+", "", t.lower()).split()

    def _distinct_choice(self, candidates: List[str]) -> str:
        recent_tokens = set()
        for rt in self._recent_feedback[-10:]:
            recent_tokens.update(w for w in self._normalize_text(rt) if w not in self.STOPWORDS)

        best = None
        best_score = 10**9
        for c in candidates:
            toks = set(w for w in self._normalize_text(c) if w not in self.STOPWORDS)
            overlap = len(toks & recent_tokens)
            if overlap < best_score:
                best, best_score = c, overlap

        return best or (candidates[0] if candidates else "")

    def _one_sentence(self, text: str) -> str:
        text = (text or "").strip().strip('"\'')

        if text.count(".") > 1 or "\n" in text:
            text = text.split("\n")[0].split(".")[0].strip() + "."
        elif not text.endswith("."):
            text += "."
        return text

    def _llm_feedback(self, rec: Dict[str, str]) -> str:
        if not self.openai:
            starter = random.choice(self.STARTERS)
            return (
                f"{starter} reviewing {rec['recommended_topic']} will strengthen the skills "
                f"you need to succeed in {rec['recommend_for']}."
            )

        style = random.choice(self.STYLE_BANK)
        starter = random.choice(self.STARTERS)

        system_msg = (
            "You write one-sentence motivational pep talks for students. "
            "Vary word choice across requests. Avoid repeating the same verbs and phrases. "
            "No greetings, no disclaimers, no definitions. One sentence only."
        )
        user_msg = (
            f"{style} Start like '{starter}' but vary it naturally. "
            f"The student struggles with {rec['recommend_for']}. "
            f"Encourage reviewing {rec['recommended_topic']} and name how it helps with {rec['recommend_for']}. "
            "Avoid generic filler. No lists, no quotes, no emojis."
        )

        try:
            r = self.openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.95,
                top_p=0.9,
                presence_penalty=0.4,
                frequency_penalty=0.4,
                max_tokens=60,
                n=3,
            )
            candidates = [self._one_sentence(choice.message.content or "") for choice in r.choices]
            best = self._distinct_choice(candidates)
            self._recent_feedback.append(best)
            return best
        except Exception:
            fallback = (
                f"{starter} reviewing {rec['recommended_topic']} will strengthen the skills "
                f"you need to succeed in {rec['recommend_for']}."
            )
            self._recent_feedback.append(fallback)
            return fallback

    def _add_llm_feedback(self, recommendations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        seen_lines = set()
        for rec in recommendations:
            line = self._llm_feedback(rec)
            if line in seen_lines:
                line = line.rstrip(".") + " — and you’re closer than you think!"
            seen_lines.add(line)
            out.append({**rec, "feedback": line})
        return out
