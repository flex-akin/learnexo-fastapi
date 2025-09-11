import os, getpass
import pandas as pd

# Paste your OpenAI key (starts with sk-...)
os.environ["OPENAI_API_KEY"] = getpass.getpass("Paste your OPENAI_API_KEY: ")

# Pick a model your account has access to
OPENAI_MODEL = "gpt-4o-mini"   # good quality + cheaper; fallback below if unavailable
# OPENAI_MODEL = "gpt-3.5-turbo"  # uncomment if you need this one

print("Key present:", "OPENAI_API_KEY" in os.environ, "| Model:", OPENAI_MODEL)

# ---- cell separator ----

from openai import OpenAI
client = OpenAI()

try:
    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"user","content":"Say OK in one word."}],
        max_tokens=5,
        temperature=0.2
    )
    print("OpenAI OK →", r.choices[0].message.content.strip())
except Exception as e:
    print("OpenAI test failed →", e)

# ---- cell separator ----


# make this guy a config
topics = [
    {"topic_id": "t1", "name": "Basic Arithmetic",     "tags": ["math", "arithmetic"], "prerequisite": None},
    {"topic_id": "t2", "name": "Fractions",            "tags": ["math", "arithmetic"], "prerequisite": "t1"},
    {"topic_id": "t3", "name": "Linear Equations",     "tags": ["math", "algebra"],    "prerequisite": "t2"},
    {"topic_id": "t4", "name": "Quadratic Equations",  "tags": ["math", "algebra"],    "prerequisite": "t3"},
    {"topic_id": "t5", "name": "Geometry Basics",      "tags": ["math", "geometry"],   "prerequisite": None},
    {"topic_id": "t6", "name": "Shapes and Angles",    "tags": ["math", "geometry"],   "prerequisite": "t5"},
]

# Example one-student scores (edit freely)
student = "student_1"
scores = {
    "Basic Arithmetic": 48,
    "Fractions": 64,
    "Linear Equations": 43,
    "Quadratic Equations": 79,
    "Geometry Basics": 63,
    "Shapes and Angles": 55,
}

# Build DataFrame expected by the recommender
name_to_id = {t["name"]: t["topic_id"] for t in topics}
rows = [{"student": student, "topic_id": name_to_id[name], "topic_name": name, "score": score}
        for name, score in scores.items()]
df = pd.DataFrame(rows)
df

# ---- cell separator ----

def get_recommendations(student, df, topics, mastery_threshold=70):
    topic_dict = {t['topic_id']: t for t in topics}
    recommendations = []

    weak_topics = df[(df['student'] == student) & (df['score'] < mastery_threshold)]

    for _, weak_row in weak_topics.iterrows():
        topic_id = weak_row['topic_id']
        topic_info = topic_dict[topic_id]
        prereq_id = topic_info.get('prerequisite')

        # 1) If a prerequisite exists, recommend it (gives a next step even if prereq is already strong)
        if prereq_id:
            prereq_name = topic_dict[prereq_id]['name']
            recommendations.append({
                "recommend_for": topic_info['name'],
                "recommended_topic": prereq_name
            })
        else:
            # 2) Otherwise recommend related topics by specific tag (e.g., 'algebra', 'geometry'),
            #    still within 'math', and not the same topic.
            specific_tags = [tag for tag in topic_info['tags'] if tag != 'math']
            for specific_tag in specific_tags:
                for t in topics:
                    if (t['topic_id'] != topic_id and
                        specific_tag in t['tags'] and
                        'math' in t['tags']):
                        recommendations.append({
                            "recommend_for": topic_info['name'],
                            "recommended_topic": t['name']
                        })

    # Deduplicate
    seen = set()
    deduped = []
    for rec in recommendations:
        key = (rec['recommend_for'], rec['recommended_topic'])
        if key not in seen:
            deduped.append(rec)
            seen.add(key)

    return deduped, []  # (core, extra)

# ---- cell separator ----

core_recs, _ = get_recommendations(student, df, topics)
core_recs

# ---- cell separator ----

def get_llm_feedback_openai(rec, model=OPENAI_MODEL):
    system_msg = (
        "You write short motivational pep talks for students. "
        "One sentence only. No introductions like 'Sure'. No definitions. "
        "Be specific and encouraging."
    )
    user_msg = (
        f"Student is struggling with {rec['recommend_for']}. "
        f"Encourage them to review {rec['recommended_topic']} and say how that helps with {rec['recommend_for']}. "
        "One sentence. Start with encouragement."
    )

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":system_msg},
                {"role":"user","content":user_msg}
            ],
            temperature=0.7,
            max_tokens=60,
        )
        text = r.choices[0].message.content.strip()
        # Safety trims: keep it one sentence and remove quote wrappers if any
        text = text.strip('"\'')

        # If it echoed instructions for some reason, nudge a fallback
        if "One sentence" in text or "encourage" in text.lower() and "review" not in text.lower():
            text = "You've got this—reviewing " + rec['recommended_topic'] + \
                   f" will sharpen the skills you need to conquer {rec['recommend_for']}!"
        return text
    except Exception as e:
        # Fallback line if API fails
        return f"(LLM error) Reviewing {rec['recommended_topic']} strengthens the building blocks for {rec['recommend_for']}—you can do it!"

# ---- cell separator ----

def add_llm_feedback(recommendations):
    enriched = []
    for rec in recommendations:
        fb = get_llm_feedback_openai(rec)
        enriched.append({**rec, "feedback": fb})
    return enriched

core_with_feedback = add_llm_feedback(core_recs)

for r in core_with_feedback:
    print(f"For improvement in {r['recommend_for']}, recommend: {r['recommended_topic']}")
    print("Feedback:", r['feedback'])
    print()

# ---- cell separator ----

import re, random

def normalize_text(t):
    return re.sub(r"[^a-z ]+", "", t.lower()).split()

def distinct_choice(candidates, recent_texts, stopwords=set(("the","a","an","and","to","of","in","for","with","on","at","by","it","is","you","your"))):
    # Choose the candidate with the least overlap vs recent outputs
    best = None
    best_score = 1e9
    recent_tokens = set()
    for rt in recent_texts[-10:]:
        recent_tokens.update(w for w in normalize_text(rt) if w not in stopwords)
    for c in candidates:
        toks = set(w for w in normalize_text(c) if w not in stopwords)
        overlap = len(toks & recent_tokens)
        score = overlap
        if score < best_score:
            best, best_score = c, score
    return best or (candidates[0] if candidates else "")

# ---- cell separator ----

RECENT_FEEDBACK = []  # append each final feedback string here so variety increases over time

# ---- cell separator ----

from openai import OpenAI
client = OpenAI()

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

def get_llm_feedback_openai(rec, model=OPENAI_MODEL):
    # Randomize some stylistic constraints to avoid repetition
    style = random.choice(STYLE_BANK)
    starter = random.choice(STARTERS)

    system_msg = (
        "You write one-sentence motivational pep talks for students. "
        "Vary word choice across requests. Avoid repeating the same verbs and phrases. "
        "No greetings, no disclaimers, no definitions. "
        "One sentence only (no conjunction chains)."
    )

    user_msg = (
        f"{style} Start with something like '{starter}' but vary it naturally. "
        f"The student struggles with {rec['recommend_for']}. "
        f"Encourage reviewing {rec['recommended_topic']} and name how it helps with {rec['recommend_for']}. "
        "Avoid generic filler (e.g., 'practice makes perfect'). "
        "No lists, no quotes, no emojis."
    )

    try:
        # Generate multiple candidates and pick a distinct one
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":system_msg},
                {"role":"user","content":user_msg},
            ],
            temperature=0.95,      # more creative
            top_p=0.9,             # nucleus sampling
            presence_penalty=0.4,  # encourage new ideas
            frequency_penalty=0.4, # reduce repetition
            max_tokens=60,
            n=3                    # get several to choose from
        )
        candidates = []
        for choice in r.choices:
            text = (choice.message.content or "").strip().strip('"\'')

            # Enforce single-sentence fallback if model drifted
            if text.count(".") > 1 or "\n" in text:
                text = text.split("\n")[0].split(".")[0].strip() + "."

            candidates.append(text)

        best = distinct_choice(candidates, RECENT_FEEDBACK)
        RECENT_FEEDBACK.append(best)
        return best

    except Exception as e:
        # Fallback with some variation
        fallback = f"{starter} reviewing {rec['recommended_topic']} sharpens the core skills you need to nail {rec['recommend_for']}."
        RECENT_FEEDBACK.append(fallback)
        return fallback

# ---- cell separator ----

def add_llm_feedback(recommendations):
    seen_lines = set()
    enriched = []
    for rec in recommendations:
        line = get_llm_feedback_openai(rec)
        # de-dup if identical line was just produced
        if line in seen_lines:
            line = line.rstrip(".") + " — and you’re closer than you think!"
        seen_lines.add(line)
        enriched.append({**rec, "feedback": line})
    return enriched

# ---- cell separator ----

core_with_feedback = add_llm_feedback(core_recs)

for r in core_with_feedback:
    print(f"For improvement in {r['recommend_for']}, recommend: {r['recommended_topic']}")
    print("Feedback:", r['feedback'])
    print()