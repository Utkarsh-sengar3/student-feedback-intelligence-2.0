import os
import re
import threading
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


SENTIMENT_MODEL_PATH = Path(
    os.getenv(
        "SENTIMENT_MODEL_PATH",
        ROOT / "models" / "sentiment" / "phase10_transformer",
    )
)

CATEGORY_MODEL_PATH = Path(
    os.getenv(
        "CATEGORY_MODEL_PATH",
        ROOT / "models" / "aspects" / "phase12_category_classifier.joblib",
    )
)

PHASE16 = ROOT / "data" / "processed" / "edurabsa" / "phase16"
PHASE17 = ROOT / "data" / "processed" / "edurabsa" / "phase17"


class FeedbackService:

    def __init__(self):

        self.sentiment_model = None
        self.sentiment_tokenizer = None

        self.category_bundle = None
        self.category_model = None
        self.label_binarizer = None
        self.categories = []

        self.actionable_insights = None
        self.dashboard_evidence = None
        self.category_recommendations = None

        self.retriever = None

        # Lazy-loading state
        self._load_lock = threading.Lock()
        self._models_load_attempted = False
        self._retriever_load_attempted = False

        self._load_phase16_data()

    def warmup(self):
        """Warm up all ML models in a background thread so inference is instant."""
        try:
            self._ensure_sentiment_model_loaded()
            self._ensure_category_model_loaded()
            self._ensure_retriever_loaded()
            from api.routes.assistant import get_rag
            get_rag()
            print("FeedbackService background warmup complete. All models ready.")
        except Exception as exc:
            print(f"FeedbackService warmup notice: {exc}")

    def _ensure_sentiment_model_loaded(self):
        """Load only the sentiment model when sentiment prediction needs it."""
        if self.sentiment_model is not None:
            return

        with self._load_lock:
            if self.sentiment_model is not None:
                return

            if not SENTIMENT_MODEL_PATH.exists():
                print(
                    f"WARNING: Sentiment model not found: "
                    f"{SENTIMENT_MODEL_PATH}"
                )
                return

            try:
                from transformers import (
                    AutoTokenizer,
                    AutoModelForSequenceClassification,
                )

                self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
                    str(SENTIMENT_MODEL_PATH),
                    local_files_only=True,
                )

                self.sentiment_model = (
                    AutoModelForSequenceClassification.from_pretrained(
                        str(SENTIMENT_MODEL_PATH),
                        local_files_only=True,
                    )
                )

                self.sentiment_model.eval()
                print("Sentiment model loaded.")

            except Exception as exc:
                self.sentiment_model = None
                self.sentiment_tokenizer = None
                print(f"WARNING: Failed to load sentiment model: {exc}")

    def _ensure_category_model_loaded(self):
        """Load only the category classifier when category prediction needs it."""
        if (
            self.category_model is not None
            and self.label_binarizer is not None
        ):
            return

        with self._load_lock:
            if (
                self.category_model is not None
                and self.label_binarizer is not None
            ):
                return

            if not CATEGORY_MODEL_PATH.exists():
                print(
                    f"WARNING: Category model not found: "
                    f"{CATEGORY_MODEL_PATH}"
                )
                return

            try:
                bundle = joblib.load(CATEGORY_MODEL_PATH)

                if not isinstance(bundle, dict):
                    print(
                        "WARNING: Phase 12 artifact is not a dictionary."
                    )
                    return

                category_model = bundle.get("model")
                label_binarizer = bundle.get("label_binarizer")

                if category_model is None or label_binarizer is None:
                    print(
                        "WARNING: Category model artifact is incomplete."
                    )
                    return

                self.category_bundle = bundle
                self.category_model = category_model
                self.label_binarizer = label_binarizer
                self.categories = list(bundle.get("categories", []))

                print("Category multilabel model loaded.")

            except Exception as exc:
                print(f"WARNING: Failed to load category model: {exc}")

    # ============================================================
    # MODEL LOADING
    # ============================================================

    def _load_models(self):

        # --------------------------------------------------------
        # DistilBERT sentiment model
        # --------------------------------------------------------

        if SENTIMENT_MODEL_PATH.exists():

            try:

                from transformers import (
                    AutoTokenizer,
                    AutoModelForSequenceClassification,
                )

                self.sentiment_tokenizer = (
                    AutoTokenizer.from_pretrained(
                        str(SENTIMENT_MODEL_PATH)
                    )
                )

                self.sentiment_model = (
                    AutoModelForSequenceClassification.from_pretrained(
                        str(SENTIMENT_MODEL_PATH)
                    )
                )

                self.sentiment_model.eval()

                print("Sentiment model loaded.")

            except Exception as exc:

                print(
                    f"WARNING: Failed to load DistilBERT: {exc}"
                )

        else:

            print(
                f"WARNING: Sentiment model not found: "
                f"{SENTIMENT_MODEL_PATH}"
            )

        # --------------------------------------------------------
        # Phase 12 multilabel category classifier
        # --------------------------------------------------------

        if CATEGORY_MODEL_PATH.exists():

            try:

                self.category_bundle = joblib.load(
                    CATEGORY_MODEL_PATH
                )

                if isinstance(self.category_bundle, dict):

                    self.category_model = (
                        self.category_bundle.get("model")
                    )

                    self.label_binarizer = (
                        self.category_bundle.get(
                            "label_binarizer"
                        )
                    )

                    self.categories = list(
                        self.category_bundle.get(
                            "categories",
                            []
                        )
                    )

                else:

                    print(
                        "WARNING: Phase 12 artifact "
                        "is not a dictionary."
                    )

                print(
                    "Category multilabel model loaded."
                )

            except Exception as exc:

                print(
                    "WARNING: Failed to load "
                    f"category model: {exc}"
                )

        else:

            print(
                f"WARNING: Category model not found: "
                f"{CATEGORY_MODEL_PATH}"
            )

    # ============================================================
    # PHASE 16 DATA
    # ============================================================

    def _ensure_retriever_loaded(self):
        """Initialize semantic retrieval only when evidence is needed."""
        if self._retriever_load_attempted:
            return

        with self._load_lock:
            if self._retriever_load_attempted:
                return
            self._retriever_load_attempted = True
            self._load_retriever()

    def _load_phase16_data(self):

        # --------------------------------------------------------
        # Actionable insights
        # --------------------------------------------------------

        actionable_path = (
            PHASE16 / "actionable_insights.csv"
        )

        if actionable_path.exists():

            try:

                self.actionable_insights = pd.read_csv(
                    actionable_path
                )

                print(
                    "Loaded actionable insights:",
                    len(self.actionable_insights)
                )

            except Exception as exc:

                print(
                    "WARNING: Failed to load "
                    f"actionable insights: {exc}"
                )

        # --------------------------------------------------------
        # Dashboard evidence package
        # --------------------------------------------------------

        evidence_path = (
            PHASE16 / "dashboard_evidence_package.csv"
        )

        if evidence_path.exists():

            try:

                self.dashboard_evidence = pd.read_csv(
                    evidence_path
                )

                print(
                    "Loaded dashboard evidence:",
                    len(self.dashboard_evidence)
                )

            except Exception as exc:

                print(
                    "WARNING: Failed to load "
                    f"dashboard evidence: {exc}"
                )

        # --------------------------------------------------------
        # Category recommendations
        # --------------------------------------------------------

        recommendations_path = (
            PHASE16 / "category_recommendations.csv"
        )

        if recommendations_path.exists():

            try:

                self.category_recommendations = (
                    pd.read_csv(
                        recommendations_path
                    )
                )

                print(
                    "Loaded category recommendations:",
                    len(self.category_recommendations)
                )

            except Exception as exc:

                print(
                    "WARNING: Failed to load "
                    f"category recommendations: {exc}"
                )

    # ============================================================
    # SEMANTIC RETRIEVER
    # ============================================================

    def _load_retriever(self):

        try:

            from src.retrieval.retriever import (
                get_shared_retriever
            )

            embeddings_path = (
                PHASE17 / "review_embeddings.npy"
            )

            metadata_path = (
                PHASE17 / "review_metadata.csv"
            )

            if (
                embeddings_path.exists()
                and metadata_path.exists()
            ):

                self.retriever = get_shared_retriever(
                    embeddings_path=embeddings_path,
                    metadata_path=metadata_path,
                    model_name="all-MiniLM-L6-v2",
                )

                print(
                    "Semantic retriever loaded."
                )

        except Exception as exc:

            print(
                "WARNING: Failed to load semantic "
                f"retriever: {exc}"
            )

    # ============================================================
    # SENTIMENT
    # ============================================================

    def _predict_sentiment(self, text):

        if self.sentiment_model is None:
            self._ensure_sentiment_model_loaded()

        if self.sentiment_model is None:
            return "Unavailable", None

        import torch

        inputs = self.sentiment_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():

            outputs = self.sentiment_model(
                **inputs
            )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        prediction = int(
            torch.argmax(
                probabilities
            ).item()
        )

        confidence = float(
            probabilities[
                prediction
            ].item()
        )

        id2label = (
            self.sentiment_model.config.id2label
        )

        label = id2label.get(
            prediction,
            str(prediction),
        )

        label = str(
            label
        ).strip().lower()

        if label in {"positive", "pos"}:

            sentiment = "Positive"

        elif label in {"negative", "neg"}:

            sentiment = "Negative"

        elif label in {"neutral", "neu"}:

            sentiment = "Neutral"

        else:

            sentiment = label.title()

        return sentiment, confidence

    # ============================================================
    # MULTILABEL CATEGORY PREDICTION
    # ============================================================

    def _predict_categories(self, text):

        if (
            self.category_model is None
            or self.label_binarizer is None
        ):
            self._ensure_category_model_loaded()

        if (
            self.category_model is None
            or self.label_binarizer is None
        ):
            return []

        try:

            prediction = (
                self.category_model.predict(
                    [text]
                )
            )

            labels = (
                self.label_binarizer.inverse_transform(
                    prediction
                )
            )

            if not labels:

                return []

            result = labels[0]

            return [
                str(label)
                for label in result
                if label is not None
            ]

        except Exception as exc:

            print(
                "Category prediction warning:",
                exc,
            )

            return []

    # ============================================================
    # PRIMARY CATEGORY
    # ============================================================
    @staticmethod
    def _select_primary_category(
        text,
        categories,
        aspect=None,
    ):
        """
        Select the most meaningful administrator-facing category.

        The multilabel classifier may return categories from different
        groups. We use the actual feedback text and detected aspect to
        choose the category that best represents the main issue.
        """

        if not categories:
            return None

        value = text.lower()

        category_rules = [
            (
                "University - Information & services",
                [
                    "information service",
                    "information services",
                    "student service",
                    "student services",
                    "administrative service",
                    "administrative services",
                    "information desk",
                    "student help",
                    "university help",
                    "university office",
                    "administration",
                    "administrative",
                    "registration office",
                    "registrar",
                    "admissions office",
                    "financial aid",
                ],
            ),
            (
                "University - Campus & facilities",
                [
                    "campus",
                    "library",
                    "cafeteria",
                    "facility",
                    "facilities",
                    "building",
                    "classroom",
                    "dorm",
                    "parking",
                    "gym",
                ],
            ),
            (
                "University - Technology & tools",
                [
                    "university portal",
                    "student portal",
                    "campus wifi",
                    "university wifi",
                    "campus internet",
                    "university website",
                ],
            ),
            (
                "Course - Workload",
                [
                    "workload",
                    "work load",
                    "too much work",
                    "heavy workload",
                    "tons of work",
                    "too many assignments",
                ],
            ),
            (
                "Course - Assessment",
                [
                    "exam",
                    "exams",
                    "midterm",
                    "final exam",
                    "test",
                    "tests",
                    "quiz",
                    "quizzes",
                    "grading",
                    "grade",
                    "assignment",
                    "assignments",
                ],
            ),
            (
                "Course - Course materials",
                [
                    "course material",
                    "course materials",
                    "slides",
                    "powerpoint",
                    "notes",
                    "textbook",
                    "readings",
                ],
            ),
            (
                "Course - Difficulty",
                [
                    "difficult",
                    "difficulty",
                    "hard",
                    "challenging",
                ],
            ),
            (
                "Course - Learning activity",
                [
                    "lecture",
                    "lectures",
                    "tutorial",
                    "tutorials",
                    "class discussion",
                    "learning activity",
                ],
            ),
            (
                "Staff - Helpfulness",
                [
                    "helpful professor",
                    "helpful teacher",
                    "helpful instructor",
                    "helpful lecturer",
                    "office hours",
                    "help students",
                ],
            ),
            (
                "Staff - Teaching",
                [
                    "teaches",
                    "teaching",
                    "explains",
                    "explain",
                    "professor",
                    "teacher",
                    "instructor",
                    "lecturer",
                ],
            ),
            (
                "Staff - Attitude",
                [
                    "rude professor",
                    "rude teacher",
                    "unfair professor",
                    "unfair teacher",
                    "disrespectful",
                    "attitude",
                ],
            ),
        ]

        candidate_scores = {}

        for category, keywords in category_rules:
            hits = sum(
                1
                for keyword in keywords
                if keyword in value
            )

            if hits > 0:
                # ML predicted categories receive higher weight; strong text signals (>= 2 hits) also candidate
                if category in categories or hits >= 2:
                    weight = 2 if category in categories else 1
                    candidate_scores[category] = hits * weight

        # Give the detected aspect extra influence.
        if aspect:

            aspect_map = {
                "workload": "Course - Workload",
                "assignment": "Course - Assessment",
                "assessment": "Course - Assessment",
                "exam": "Course - Assessment",
                "course materials": "Course - Course materials",
                "difficulty": "Course - Difficulty",
                "teaching": "Staff - Teaching",
                "technology": "University - Technology & tools",
                "campus facilities": "University - Campus & facilities",
                "student services": "University - Information & services",
                "information & services": "University - Information & services",
            }

            mapped = aspect_map.get(
                str(aspect).strip().lower()
            )

            if mapped:
                candidate_scores[mapped] = (
                    candidate_scores.get(mapped, 0) + (3 if mapped in categories else 2)
                )

        if candidate_scores:

            # If university feedback explicitly concerns technology,
            # prefer the technology category when the classifier
            # has already identified a university-related concern.
            university_categories = {
                "University - Information & services",
                "University - Campus & facilities",
                "University - Technology & tools",
            }

            predicted_university_categories = [
                category
                for category in categories
                if category in university_categories
            ]

            technology_keywords = [
                "wifi",
                "wi-fi",
                "internet",
                "network connection",
                "university portal",
                "student portal",
                "technical issue",
                "technology",
            ]

            has_technology_signal = any(
                keyword in value
                for keyword in technology_keywords
            )

            if (
                has_technology_signal
                and predicted_university_categories
            ):
                return "University - Technology & tools"

            return max(
                candidate_scores,
                key=candidate_scores.get,
            )

        # Fall back to specific categories over generic Overall.
        specific = [
            category
            for category in categories
            if not category.endswith(
                " - Overall"
            )
        ]

        if specific:
            return specific[0]

        return categories[0]

    # ============================================================
    # ASPECT EXTRACTION
    # ============================================================

    @classmethod
    def _extract_aspects(cls, text, categories=None):
        """
        Extract all relevant aspects mentioned in student feedback.

        Supports detection of:
        - teaching (professor, teacher, lecture, delivery)
        - exam (midterm, final exam, test, quiz, grading)
        - assignment (projects, homework, essays, lab reports)
        - workload (heavy workload, too much work, pace)
        - course materials (slides, textbook, readings, notes)
        - technology (wi-fi, internet, student portal, connectivity)
        - campus facilities (library, cafeteria, classrooms, parking, dorm)
        - student services (administrative office, admissions, financial aid)
        - difficulty (challenging, difficult, complicated)

        Returns:
            list[str]: Detected aspects in order of relevance.
        """
        value = str(text or "").lower()
        categories = categories or []

        keyword_map = [
            (
                "technology",
                [
                    "wi-fi",
                    "wifi",
                    "wireless internet",
                    "internet connection",
                    "internet",
                    "network connection",
                    "network",
                    "online portal",
                    "student portal",
                    "university portal",
                    "technical issue",
                    "technical issues",
                    "canvas",
                    "blackboard",
                    "moodle",
                    "technology",
                ],
            ),
            (
                "student services",
                [
                    "student services",
                    "student service",
                    "student support",
                    "financial aid",
                    "registration office",
                    "registration",
                    "registrar",
                    "admissions office",
                    "admissions",
                    "administrative office",
                    "university office",
                    "support services",
                ],
            ),
            (
                "campus facilities",
                [
                    "campus facilities",
                    "campus facility",
                    "campus",
                    "library",
                    "cafeteria",
                    "facilities",
                    "facility",
                    "parking",
                    "dormitory",
                    "dorm",
                    "gym",
                    "building maintenance",
                    "classroom facilities",
                    "classrooms",
                ],
            ),
            (
                "workload",
                [
                    "workload",
                    "work load",
                    "too much work",
                    "heavy workload",
                    "heavy work",
                    "tons of work",
                    "overwhelming workload",
                ],
            ),
            (
                "assignment",
                [
                    "assignments",
                    "assignment",
                    "essays",
                    "essay",
                    "projects",
                    "project",
                    "homework",
                    "lab report",
                    "lab reports",
                ],
            ),
            (
                "exam",
                [
                    "final exam",
                    "midterm",
                    "midterms",
                    "exams",
                    "exam",
                    "tests",
                    "test",
                    "quiz",
                    "quizzes",
                    "grading",
                    "marking",
                ],
            ),
            (
                "course materials",
                [
                    "course materials",
                    "course material",
                    "lecture notes",
                    "slides",
                    "powerpoint",
                    "readings",
                    "textbook",
                    "course notes",
                ],
            ),
            (
                "teaching",
                [
                    "professor",
                    "instructor",
                    "lecturer",
                    "teacher",
                    "teaching",
                    "lecture",
                    "lectures",
                    "explains",
                    "explanation",
                ],
            ),
            (
                "difficulty",
                [
                    "difficult",
                    "difficulty",
                    "hard",
                    "challenging",
                    "too complicated",
                ],
            ),
        ]

        detected = []
        for aspect, keywords in keyword_map:
            if any(keyword in value for keyword in keywords):
                if aspect not in detected:
                    detected.append(aspect)

        # Fallback to category hints if no text keyword was matched
        if not detected and categories:
            category_aspect_map = [
                ("University - Technology & tools", "technology"),
                ("University - Campus & facilities", "campus facilities"),
                ("University - Information & services", "student services"),
            ]

            category_strings = [
                str(category) for category in categories if category is not None
            ]

            for category_name, aspect in category_aspect_map:
                if category_name in category_strings and aspect not in detected:
                    detected.append(aspect)

            for category in category_strings:
                category_lower = category.lower()
                if "workload" in category_lower and "workload" not in detected:
                    detected.append("workload")
                elif "assessment" in category_lower and "exam" not in detected:
                    detected.append("exam")
                elif "course materials" in category_lower and "course materials" not in detected:
                    detected.append("course materials")
                elif "difficulty" in category_lower and "difficulty" not in detected:
                    detected.append("difficulty")
                elif ("teaching" in category_lower or "attitude" in category_lower) and "teaching" not in detected:
                    detected.append("teaching")

        return detected

    @classmethod
    def _extract_aspect(cls, text, categories):
        """
        Extract the primary aspect from student feedback (for backward compatibility).
        """
        aspects = cls._extract_aspects(text, categories)
        return aspects[0] if aspects else None

    # ============================================================
    # SEVERITY
    # ============================================================

    @staticmethod
    def _severity(
        text,
        sentiment,
    ):

        if sentiment != "Negative":

            return "Low"

        value = text.lower()

        critical_terms = [
            "harassment",
            "unsafe",
            "abuse",
            "racist",
            "threat",
            "danger",
            "fraud",
            "corrupt",
            "sexual",
        ]

        high_terms = [
            "unfair",
            "bias",
            "bully",
            "discrimination",
            "inappropriate",
            "serious",
            "extremely",
            "unacceptable",
            "ridiculous",
            "unreasonable",
        ]

        medium_terms = [
            "poor",
            "bad",
            "terrible",
            "difficult",
            "confusing",
            "unclear",
            "missing",
            "slow",
            "late",
            "unhelpful",
            "problem",
            "issue",
            "weak",
        ]

        if any(
            term in value
            for term in critical_terms
        ):

            return "Critical"

        if any(
            term in value
            for term in high_terms
        ):

            return "High"

        if any(
            term in value
            for term in medium_terms
        ):

            return "Medium"

        return "Low"

    # ============================================================
    # CATEGORY AND ASPECT-AWARE RECOMMENDATIONS
    # ============================================================

    def _get_category_recommendation(
        self,
        category,
        aspect=None,
        text=None,
        sentiment=None,
    ):
        """
        Return a recommendation grounded in the predicted category
        and the specific aspect mentioned in the feedback.

        Existing dataset recommendations remain the fallback.
        """

        if category is None:
            return None

        category = str(category).strip()
        aspect = str(aspect or "").strip().lower()
        text_lower = str(text or "").strip().lower()
        sentiment_lower = str(sentiment or "").strip().lower()

        # --------------------------------------------------------
        # Positive feedback: acknowledge rather than prescribe
        # corrective action.
        # --------------------------------------------------------

        if sentiment_lower == "positive":
            return {
                "recommended_action": (
                    "Preserve the practices highlighted in this "
                    "positive feedback and review whether they can "
                    "be maintained or shared across other courses "
                    "and student services."
                ),
                "recommended_owner": "Relevant academic or service team",
            }

        # --------------------------------------------------------
        # Aspect-aware university recommendations
        # --------------------------------------------------------

        if category.startswith("University -"):

            if aspect == "technology" or any(
                term in text_lower
                for term in (
                    "wi-fi",
                    "wifi",
                    "internet connection",
                    "network connection",
                    "connectivity",
                )
            ):
                return {
                    "recommended_action": (
                        "Investigate Wi-Fi reliability, network "
                        "coverage, authentication failures, and "
                        "connection interruptions. Review outage "
                        "logs and affected locations, then validate "
                        "the fix with students."
                    ),
                    "recommended_owner": (
                        "IT Services / Network Administration"
                    ),
                }

            if aspect == "student services" or any(
                term in text_lower
                for term in (
                    "student services",
                    "student service",
                    "administration",
                    "registration support",
                    "admissions",
                    "financial aid",
                )
            ):
                return {
                    "recommended_action": (
                        "Review the student-service process for "
                        "response delays, unclear instructions, and "
                        "repeated student enquiries. Define response "
                        "time targets and provide clear escalation "
                        "routes."
                    ),
                    "recommended_owner": (
                        "Student Services / Administration"
                    ),
                }

            if aspect == "campus facilities" or any(
                term in text_lower
                for term in (
                    "campus",
                    "library",
                    "cafeteria",
                    "facilities",
                    "facility",
                    "classroom",
                    "parking",
                )
            ):
                return {
                    "recommended_action": (
                        "Inspect the reported campus facility, "
                        "identify maintenance or accessibility "
                        "problems, and prioritize repairs using "
                        "student impact and verified evidence."
                    ),
                    "recommended_owner": (
                        "Campus Facilities / Maintenance"
                    ),
                }

            return {
                "recommended_action": (
                    "Review the relevant university service or "
                    "facility using the feedback and supporting "
                    "evidence. Identify the specific operational "
                    "issue before selecting an intervention."
                ),
                "recommended_owner": (
                    "Relevant University Department"
                ),
            }

        # --------------------------------------------------------
        # Aspect-aware course recommendations
        # --------------------------------------------------------

        if category.startswith("Course -"):

            if aspect == "exam":
                action = (
                    "Review examination difficulty, question "
                    "clarity, syllabus coverage, and alignment "
                    "with taught learning objectives. Compare "
                    "the reported concern with the assessment "
                    "design and student evidence."
                )

            elif aspect == "assignment":
                action = (
                    "Review assignment instructions, workload, "
                    "marking criteria, deadlines, and alignment "
                    "with course learning objectives."
                )

            elif aspect == "workload":
                action = (
                    "Review the distribution of coursework, "
                    "deadlines, and estimated completion time "
                    "across the course."
                )

            elif aspect == "course materials":
                action = (
                    "Review the availability, accuracy, clarity, "
                    "and timeliness of course materials and "
                    "provide missing or updated resources."
                )

            elif aspect == "difficulty":
                action = (
                    "Review the reported difficulty against "
                    "prerequisite knowledge, teaching coverage, "
                    "practice opportunities, and assessment "
                    "expectations."
                )

            else:
                action = (
                    "Review the reported course concern against "
                    "learning objectives, course delivery, and "
                    "supporting student feedback."
                )

            return {
                "recommended_action": action,
                "recommended_owner": (
                    "Course Instructor / Academic Coordinator"
                ),
            }

        # --------------------------------------------------------
        # Aspect-aware staff recommendations
        # --------------------------------------------------------

        if category.startswith("Staff -"):

            if aspect == "teaching":
                action = (
                    "Review teaching clarity, pacing, explanation "
                    "quality, and classroom delivery. Use specific "
                    "feedback examples to identify opportunities "
                    "for teaching improvement."
                )

            elif aspect == "student services":
                action = (
                    "Review the reported support interaction, "
                    "response time, and clarity of guidance."
                )

            else:
                action = (
                    "Review the reported staff interaction and "
                    "identify practical improvements in "
                    "communication, responsiveness, and student "
                    "support."
                )

            return {
                "recommended_action": action,
                "recommended_owner": (
                    "Department Head / Academic Coordinator"
                ),
            }

        # --------------------------------------------------------
        # Existing dataset-based recommendation fallback
        # --------------------------------------------------------

        if self.category_recommendations is None:
            return None

        df = self.category_recommendations

        if "category" not in df.columns:
            return None

        matches = df[
            df["category"].astype(str).str.strip()
            == category
        ]

        if matches.empty:
            return None

        if "volume" in matches.columns:
            matches = matches.sort_values(
                "volume",
                ascending=False,
            )

        row = matches.iloc[0]

        action = str(
            row.get("recommended_action", "")
        ).strip()

        owner = str(
            row.get("recommended_owner", "")
        ).strip()

        if not action:
            return None

        return {
            "recommended_action": action,
            "recommended_owner": owner,
        }

    # ============================================================
    # PRIORITY / CONCEPT MATCH
    # ============================================================
    def _get_priority_data(self, categories, aspect):
        """
        Match predicted categories and extracted aspects against
        existing Phase 16 actionable-insight concepts.

        Important:
        - Uses existing dataset concepts only.
        - Does not invent priority scores.
        - Does not map unrelated concepts just to produce a score.
        - Returns None when no defensible match exists.
        """

        if self.actionable_insights is None:
            return None

        if not isinstance(self.actionable_insights, pd.DataFrame):
            return None

        df = self.actionable_insights.copy()

        required_columns = {
            "concept",
            "priority_score",
        }

        if not required_columns.issubset(df.columns):
            return None

        if df.empty:
            return None

        # Normalize inputs.
        if categories is None:
            categories = []

        categories = [
            str(category)
            for category in categories
            if category is not None
        ]

        aspect = (
            str(aspect).strip().lower()
            if aspect is not None
            else None
        )

        # Existing concepts from the Phase 16 dataset.
        available_concepts = set(
            df["concept"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        concepts = set()

        # --------------------------------------------------------
        # CATEGORY-TO-CONCEPT HINTS
        # --------------------------------------------------------

        category_concept_map = {
            "Course - Workload": [
                "course",
                "workload",
                "work load",
            ],
            "Course - Assessment": [
                "exam",
                "assignment",
                "tests",
                "test",
                "quiz",
                "assessments",
            ],
            "Course - Course materials": [
                "course materials",
                "course material",
                "readings",
                "notes",
                "textbook",
            ],
            "Course - Difficulty": [
                "difficulty",
                "course",
                "class",
            ],
            "Staff - Teaching": [
                "professor",
                "teacher",
                "teaching",
                "instructor",
                "lecturer",
            ],
            "Staff - Attitude": [
                "teacher",
                "professor",
            ],
            "University - Technology & tools": [
                "wifi",
            ],
            "University - Information & services": [
                "information",
            ],
        }

        for category in categories:
            for category_name, mapped_concepts in (
                category_concept_map.items()
            ):
                if category == category_name:
                    concepts.update(mapped_concepts)

        # --------------------------------------------------------
        # ASPECT-TO-CONCEPT HINTS
        # --------------------------------------------------------

        aspect_concept_map = {
            "workload": [
                "course",
                "workload",
                "work load",
            ],
            "assignment": [
                "assignment",
                "homework",
                "projects",
                "essays",
            ],
            "exam": [
                "exam",
                "tests",
                "test",
                "quiz",
                "final",
                "midterms",
                "midterm and final",
            ],
            "teaching": [
                "professor",
                "teacher",
                "teaching",
                "instructor",
                "lecturer",
            ],
            "course materials": [
                "course materials",
                "course material",
                "course notes",
                "lecture notes",
                "readings",
                "notes",
            ],
            "difficulty": [
                "difficulty",
                "course",
                "class",
            ],
            "technology": [
                "wifi",
            ],
            "student services": [
                "information",
            ],
        }

        if aspect in aspect_concept_map:
            concepts.update(aspect_concept_map[aspect])

        # Keep only concepts present in the actual dataset.
        concepts = {
            concept
            for concept in concepts
            if concept in available_concepts
        }

        if not concepts:
            return None

        # --------------------------------------------------------
        # MATCH AGAINST ACTIONABLE INSIGHTS
        # --------------------------------------------------------

        normalized_concepts = (
            df["concept"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        matches = df[
            normalized_concepts.isin(concepts)
        ].copy()

        if matches.empty:
            return None

        # Safely handle invalid or missing scores.
        matches["priority_score"] = pd.to_numeric(
            matches["priority_score"],
            errors="coerce",
        )

        matches = matches.dropna(
            subset=["priority_score"]
        )

        if matches.empty:
            return None

        # Use the highest-scoring matching concept.
        matches = matches.sort_values(
            "priority_score",
            ascending=False,
        )

        row = matches.iloc[0]

        # --------------------------------------------------------
        # BUILD RESPONSE
        # --------------------------------------------------------

        def safe_string(column, default=""):
            value = row.get(column, default)

            if pd.isna(value):
                return default

            return str(value)

        return {
            "concept": safe_string("concept"),
            "priority_score": float(row["priority_score"]),
            "priority": safe_string("priority"),
            "severity_level": safe_string("severity_level"),
            "recommended_action": safe_string(
                "recommended_action"
            ),
            "recommended_owner": safe_string(
                "recommended_owner"
            ),
        }
    # ============================================================
    # SEMANTIC EVIDENCE
    # ============================================================

    def _get_evidence(
    self,
    text,
    sentiment,
    categories,
    primary_category=None,
    ):
        """
        Retrieve semantically relevant evidence while respecting
        predicted domain and sentiment.

        The retriever already returns the correct metadata rows,
        so we use those rows directly instead of re-indexing
        self.retriever.metadata.
        """

        if self.retriever is None:
            self._ensure_retriever_loaded()

        if self.retriever is None:
            return []

        try:
            # ----------------------------------------------------
            # Normalize sentiment
            # ----------------------------------------------------

            normalized_sentiment = None

            if sentiment in {
                "Positive",
                "Negative",
                "Neutral",
            }:
                normalized_sentiment = sentiment.lower()

            # ----------------------------------------------------
            # Infer domain from predicted categories
            # ----------------------------------------------------

            domain_filter = None
            domains_present = set()
            for cat in (categories or []):
                cat_str = str(cat)
                if cat_str.startswith("University -"):
                    domains_present.add("university review")
                elif cat_str.startswith("Staff -"):
                    domains_present.add("teacher review")
                elif cat_str.startswith("Course -"):
                    domains_present.add("course review")

            # Only restrict domain if all detected categories belong to a single domain
            if len(domains_present) == 1:
                domain_filter = list(domains_present)[0]
            elif primary_category and not domains_present:
                primary_cat_str = str(primary_category)
                if primary_cat_str.startswith("University -"):
                    domain_filter = "university review"
                elif primary_cat_str.startswith("Staff -"):
                    domain_filter = "teacher review"
                elif primary_cat_str.startswith("Course -"):
                    domain_filter = "course review"

            # ----------------------------------------------------
            # Retrieve candidates
            # ----------------------------------------------------

            results = self.retriever.search(
                query=text,
                top_k=15,
                sentiment=normalized_sentiment,
                domain=domain_filter,
            )

            if results is None or results.empty:
                return []

            # ----------------------------------------------------
            # Category-specific precise keywords (excluding generic stopwords)
            # ----------------------------------------------------

            category_keywords = {
                "Course - Workload": [
                    "workload",
                    "work load",
                    "heavy workload",
                    "too much work",
                    "tons of work",
                    "overwhelming workload",
                    "busywork",
                ],

                "Course - Assessment": [
                    "exam",
                    "exams",
                    "midterm",
                    "midterms",
                    "final exam",
                    "quiz",
                    "quizzes",
                    "grading",
                    "rubric",
                    "marking",
                ],

                "Course - Content": [
                    "syllabus",
                    "curriculum",
                    "course content",
                    "concepts",
                    "prerequisite",
                ],

                "Course - Course materials": [
                    "slides",
                    "powerpoint",
                    "lecture notes",
                    "readings",
                    "textbook",
                    "syllabi",
                ],

                "Course - Difficulty": [
                    "difficult",
                    "difficulty",
                    "overly hard",
                    "too complicated",
                    "impossible",
                ],

                "Course - Learning activity": [
                    "tutorial",
                    "tutorials",
                    "lab session",
                    "group project",
                    "workshop",
                ],

                "Staff - Teaching": [
                    "professor",
                    "instructor",
                    "lecturer",
                    "teaching style",
                    "explanation",
                    "explains poorly",
                ],

                "Staff - Helpfulness": [
                    "helpful professor",
                    "approachable",
                    "office hours",
                    "unhelpful",
                    "available to help",
                ],

                "Staff - Attitude": [
                    "rude",
                    "attitude",
                    "disrespectful",
                    "unfair instructor",
                    "condescending",
                    "arrogant",
                ],

                "University - Information & services": [
                    "student service",
                    "student services",
                    "registration office",
                    "registrar",
                    "admissions office",
                    "financial aid office",
                    "student advisor",
                ],

                "University - Campus & facilities": [
                    "campus",
                    "library",
                    "cafeteria",
                    "facilities",
                    "dormitory",
                    "dorm",
                    "parking lot",
                    "restroom",
                    "gym",
                ],

                "University - Technology & tools": [
                    "wifi",
                    "wi-fi",
                    "wireless internet",
                    "student portal",
                    "university portal",
                    "blackboard",
                    "canvas portal",
                    "network connection",
                ],
            }

            keywords = set()

            for category in categories:
                keywords.update(
                    category_keywords.get(
                        str(category),
                        [],
                    )
                )

            # ----------------------------------------------------
            # Rerank retrieved rows
            # ----------------------------------------------------

            scored_rows = []

            for _, row in results.iterrows():

                evidence_text = row.get(
                    "text",
                    "",
                )

                if pd.isna(evidence_text):
                    evidence_text = ""

                evidence_text = str(evidence_text)

                evidence_lower = evidence_text.lower()

                similarity = float(
                    row.get(
                        "similarity_score",
                        0.0,
                    )
                )

                domain = row.get(
                    "domain",
                    None,
                )

                sentiment_value = row.get(
                    "overall_sentiment",
                    row.get(
                        "sentiment",
                        None,
                    ),
                )

                if pd.isna(domain):
                    domain = None

                if pd.isna(sentiment_value):
                    sentiment_value = None

                # ------------------------------------------------
                # Precise Category keyword matches (conservative)
                # ------------------------------------------------

                keyword_hits = sum(
                    1
                    for keyword in keywords
                    if keyword in evidence_lower
                )

                # ------------------------------------------------
                # Final score (semantic similarity dominates)
                # ------------------------------------------------

                final_score = similarity

                if keyword_hits > 0:
                    final_score += min(
                        0.04,
                        0.01 * keyword_hits,
                    )

                if (
                    domain_filter is not None
                    and str(domain).strip().lower()
                    == domain_filter
                ):
                    final_score += 0.03

                scored_rows.append(
                    {
                        "review_id": row.get(
                            "review_id"
                        ),
                        "similarity_score": similarity,
                        "evidence_score": final_score,
                        "domain": (
                            str(domain)
                            if domain is not None
                            else None
                        ),
                        "sentiment": (
                            str(sentiment_value)
                            if sentiment_value is not None
                            else None
                        ),
                        "text": evidence_text,
                    }
                )

            # ----------------------------------------------------
            # Sort
            # ----------------------------------------------------

            scored_rows.sort(
                key=lambda item: item[
                    "evidence_score"
                ],
                reverse=True,
            )

            # ----------------------------------------------------
            # Return top 5
            # ----------------------------------------------------

            return [
                {
                    "review_id": item["review_id"],
                    "similarity_score": float(
                        item["similarity_score"]
                    ),
                    "domain": item["domain"],
                    "sentiment": item["sentiment"],
                    "text": item["text"],
                }
                for item in scored_rows[:5]
            ]

        except Exception as exc:

            print(
                "Evidence retrieval warning:",
                exc,
            )

            return []
    # ============================================================
    # ANALYZE
    # ============================================================

    def analyze(
        self,
        text,
    ):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Feedback text must be a non-empty string.")

        text = re.sub(
            r"\s+",
            " ",
            text.strip(),
        )

        sentiment, confidence = (
            self._predict_sentiment(
                text
            )
        )

        categories = (
            self._predict_categories(
                text
            )
        )

        aspects = (
            self._extract_aspects(
                text,
                categories,
            )
        )
        aspect = aspects[0] if aspects else None

        primary_category = (
            self._select_primary_category(
                text,
                categories,
                aspect,
            )
        )

        # Category consistency: Ensure primary_category is included in multilabel predictions
        if primary_category and primary_category not in categories:
            categories.append(primary_category)

        severity = (
            self._severity(
                text,
                sentiment,
            )
        )

        priority_data = (
            self._get_priority_data(
                categories,
                aspect,
            )
        )

        category_recommendation = (
            self._get_category_recommendation(
                category=primary_category,
                aspect=aspect,
                text=text,
                sentiment=sentiment,
            )
        )

        evidence = self._get_evidence(
            text,
            sentiment,
            categories,
            primary_category,
        )

        # --------------------------------------------------------
        # Recommendation
        # --------------------------------------------------------

        recommendation = "Review supporting evidence before taking action."

        if category_recommendation:
            recommendation = category_recommendation["recommended_action"]

        elif priority_data and priority_data.get("recommended_action"):
            recommendation = priority_data["recommended_action"]

        # --------------------------------------------------------
        # INDIVIDUAL FEEDBACK PRIORITY
        # --------------------------------------------------------

        priority_level = None
        priority_score = None

        # --------------------------------------------------------
        # AGGREGATE CONCERN PRIORITY
        # --------------------------------------------------------

        concern_priority_level = None
        concern_priority_score = None

        if priority_data:

            concern_priority_level = (
                priority_data.get("priority")
            )

            raw_concern_score = priority_data.get(
                "priority_score"
            )

            if raw_concern_score is not None:
                concern_priority_score = float(
                    raw_concern_score
                )

        # --------------------------------------------------------
        # INDIVIDUAL FEEDBACK ACTION PRIORITY
        # --------------------------------------------------------

        normalized_sentiment = str(
            sentiment or ""
        ).strip().lower()

        normalized_severity = str(
            severity or ""
        ).strip().lower()

        # Positive feedback is not an unresolved complaint.
        if normalized_sentiment == "positive":
            priority_level = "Informational"
            priority_score = 0.0

        elif normalized_sentiment in {
            "negative",
            "neutral",
        }:

            severity_scores = {
                "low": 0.25,
                "medium": 0.50,
                "high": 0.75,
                "critical": 1.0,
            }

            priority_score = severity_scores.get(
                normalized_severity
            )

            if priority_score is not None:

                if normalized_severity == "critical":
                    priority_level = "Critical"

                elif normalized_severity == "high":
                    priority_level = "High"

                elif normalized_severity == "medium":
                    priority_level = "Medium"

                else:
                    priority_level = "Low"
        return {
            "text": text,
            "sentiment": sentiment,
            "sentiment_confidence": confidence,
            "category": primary_category,
            "categories": categories,
            "aspect": aspect,
            "aspects": aspects,
            "severity": severity,
            "priority_level": priority_level,
            "priority_score": priority_score,
            "concern_priority_level": concern_priority_level,
            "concern_priority_score": concern_priority_score,
            "recommendation": recommendation,
            "evidence": evidence,
        }

    # ============================================================
    # STATUS
    # ============================================================

    def status(self):

        return {

            "sentiment_model_loaded": (
                self.sentiment_model
                is not None
            ),

            "category_model_loaded": (
                self.category_model
                is not None
            ),

            "category_count": len(
                self.categories
            ),

            "phase16_insights_loaded": (
                self.actionable_insights
                is not None
            ),

            "phase16_evidence_loaded": (
                self.dashboard_evidence
                is not None
            ),

            "semantic_retriever_loaded": (
                self.retriever
                is not None
            ),

            "sentiment_model": (
                "phase10_transformer"
            ),

            "category_model": (
                "phase12_category_classifier"
            ),

        }