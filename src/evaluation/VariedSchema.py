{
  "domains": {
    "math": {
      "dimensions": {
        "formal_correctness": {
          "definition": "Definitions, theorems, claims, and conclusions are stated correctly (including constraints/quantifiers) with no slide-unsupported inventions.",
          "score_anchors": {
            "1": "Major definitions/results are incorrect or fabricated; frequent unsupported claims.",
            "3": "Mostly correct but with missing conditions or minor inaccuracies; few unsupported claims.",
            "5": "Accurate statements of central definitions/results with essential conditions; no unsupported claims."
          }
        },
        "assumptions_scope": {
          "definition": "Explicitly includes key assumptions/conditions and clarifies the scope where results or methods apply.",
          "score_anchors": {
            "1": "Assumptions/conditions are missing or wrong; treats conditional results as universal.",
            "3": "Some assumptions/conditions included but important ones are missing or vague.",
            "5": "Key assumptions/conditions and applicability are clearly stated and match the slides."
          }
        },
        "derivation_structure": {
          "definition": "Captures the high-level reasoning chain of proofs/derivations (main steps) without inventing steps not supported by slides.",
          "score_anchors": {
            "1": "Reasoning chain is incoherent, scrambled, or fabricated; contradicts slides.",
            "3": "Main idea present but steps are incomplete or somewhat muddled.",
            "5": "Clear, slide-consistent outline of the major steps leading to the result."
          }
        },
        "notation_precision": {
          "definition": "Uses symbols/variables/terminology consistently and correctly; avoids meaning-changing notation errors.",
          "score_anchors": {
            "1": "Frequent notation errors that change meaning; confusing symbol use.",
            "3": "Minor notation issues but meaning is mostly preserved.",
            "5": "Notation and terminology are accurate, consistent, and unambiguous."
          }
        },
        "procedural_utility": {
          "definition": "When procedures/algorithms exist, explains inputs → steps → outputs correctly and in a usable way.",
          "score_anchors": {
            "1": "Procedure is unusable/incorrect; missing key inputs/outputs or wrong steps.",
            "3": "Understandable but missing important steps/details or has ambiguity.",
            "5": "Clear and correct method recipe aligned with the slides."
          }
        },
        "limitations_edge_cases": {
          "definition": "Notes limitations, edge cases, or failure modes discussed (or clearly implied) and avoids overgeneralizing.",
          "score_anchors": {
            "1": "No limits; misleading overconfidence or broad claims beyond slides.",
            "3": "Mentions limits but superficially or incompletely.",
            "5": "Key limitations/edge cases are captured and tied to applicability."
          }
        }
      }
    },

    "humanities": {
      "dimensions": {
        "thesis_capture": {
          "definition": "Identifies the lecture’s central question/thesis and what the lecture is arguing, not just a list of topics.",
          "score_anchors": {
            "1": "No clear thesis or an incorrect one; mostly topic listing.",
            "3": "Thesis is present but generic or partially misaligned.",
            "5": "Thesis/question is explicit, accurate, and matches lecture framing."
          }
        },
        "conceptual_nuance": {
          "definition": "Preserves key concepts, definitions, and distinctions as used in the lecture (no flattening or distortion).",
          "score_anchors": {
            "1": "Key concepts are misdefined/flattened; major distinctions lost.",
            "3": "Most concepts are correct but nuance/distinctions are blurred or underexplained.",
            "5": "Concepts and distinctions are accurate and preserve intended meanings."
          }
        },
        "argument_coherence": {
          "definition": "Summarizes the reasoning chain (claim → reasoning → support) in a coherent progression that reflects how the lecture builds its argument.",
          "score_anchors": {
            "1": "Disjointed; argument flow missing or fabricated.",
            "3": "Some structure, but linkage between claims and reasoning is incomplete.",
            "5": "Clear argument flow that matches the lecture’s development and support."
          }
        },
        "evidence_grounding": {
          "definition": "Connects claims to lecture-discussed examples/texts/artifacts without fabricating quotes or references.",
          "score_anchors": {
            "1": "Examples are missing or invented; fabricated quotes/references appear.",
            "3": "Some real examples included, but linkage is weak or incomplete.",
            "5": "Examples are slide-grounded and clearly tied to the claims they support; no invention."
          }
        },
        "context_positioning": {
          "definition": "Places ideas in relevant historical/intellectual context when present (authors, movements, schools of thought, time/setting).",
          "score_anchors": {
            "1": "Context is absent or incorrect; misattributions or anachronisms.",
            "3": "Some context included but incomplete or loosely connected.",
            "5": "Correct, relevant context that supports understanding of the argument."
          }
        },
        "tension_stakes": {
          "definition": "Communicates nuance: acknowledges tensions/counterpoints/ambiguities and explains stakes/implications emphasized in the lecture.",
          "score_anchors": {
            "1": "Overly certain, ignores debate, or invents implications not present in the lecture.",
            "3": "Mentions tension/stakes but generally or without clear connection to lecture framing.",
            "5": "Accurately captures key tensions and meaningful implications grounded in the lecture."
          }
        }
      }
    },

    "natural_sciences": {
      "dimensions": {
        "mechanism_accuracy": {
          "definition": "Accurately explains the core mechanisms/processes (how/why), not just correlations.",
          "score_anchors": {
            "1": "Misses or misstates the main mechanism; invents processes not in slides.",
            "3": "Includes the main mechanism but misses key steps/variables or is shallow.",
            "5": "Captures the main mechanism with the key steps/variables emphasized on slides."
          }
        },
        "model_fidelity": {
          "definition": "Accurately represents models/diagrams/equations/frameworks (structure, variable roles, relationships) without invention.",
          "score_anchors": {
            "1": "Models/relationships are wrong or fabricated; contradicts slide representations.",
            "3": "Mostly correct but missing key structure or minor misinterpretations.",
            "5": "Models/representations are accurately described with correct roles/relationships."
          }
        },
        "evidence_logic": {
          "definition": "Explains how evidence supports claims (methods/measurements/comparisons) in a slide-consistent way.",
          "score_anchors": {
            "1": "No valid evidence-to-claim logic; claims are ungrounded or out of order.",
            "3": "Some linkage but incomplete experimental/observational logic.",
            "5": "Clear chain from evidence/methods to results to conclusions as presented."
          }
        },
        "results_vs_interpretation": {
          "definition": "Separates observed results from hypotheses/interpretation and avoids overstating certainty.",
          "score_anchors": {
            "1": "Conflates results with interpretation; misleading certainty or incorrect conclusions.",
            "3": "Some separation, but blurred in places or occasionally overstated.",
            "5": "Consistently distinguishes results vs interpretation and labels tentative claims."
          }
        },
        "uncertainty_limits": {
          "definition": "Captures uncertainty, assumptions, limitations, error sources, and generalizability constraints emphasized in the lecture.",
          "score_anchors": {
            "1": "No limits/uncertainty; overconfident or invents uncertainty details.",
            "3": "Mentions limits/uncertainty but misses important ones or stays generic.",
            "5": "Key uncertainties/limitations are clearly stated and tied to interpretation."
          }
        },
        "implications_next_steps": {
          "definition": "Connects findings/models to implications, predictions, applications, or next questions when present on slides.",
          "score_anchors": {
            "1": "Implications are missing or invented; overclaims applications.",
            "3": "Some implications but generic or loosely grounded.",
            "5": "Slide-grounded implications and next questions are clearly stated."
          }
        }
      }
    },

    "business": {
      "dimensions": {
        "decision_context": {
          "definition": "Captures the core problem/decision, objective, and key situation facts the lecture/case is built around.",
          "score_anchors": {
            "1": "Problem/objective is missing or incorrect; key context is absent.",
            "3": "Problem/objective is stated but important context or constraints are missing.",
            "5": "Clearly states the core decision/problem, objectives, and the essential context from the slides."
          }
        },
        "metrics_correctness": {
          "definition": "Uses business terms and any numbers/metrics exactly as presented; does not invent data, outcomes, or relationships.",
          "score_anchors": {
            "1": "Invents metrics/results or misstates core facts; multiple unsupported claims.",
            "3": "Mostly accurate, but has minor factual/metric errors or some overreach beyond slides.",
            "5": "Accurate and slide-grounded; metrics and claims are represented correctly with no invented facts."
          }
        },
        "analysis_structure": {
          "definition": "Organizes the content in a decision-oriented flow (context → analysis → options/tradeoffs → recommendation).",
          "score_anchors": {
            "1": "Disorganized; lacks a coherent decision flow; options/recommendation are unclear or misplaced.",
            "3": "Some structure, but analysis/options/tradeoffs are not clearly separated or sequenced.",
            "5": "Clear progression from context to analysis to options/tradeoffs to a well-placed conclusion/recommendation."
          }
        },
        "actionability": {
          "definition": "Makes the reasoning actionable: clearly explains why conclusions follow and what actions/options are being considered.",
          "score_anchors": {
            "1": "Vague takeaways; hard to tell what to do or why; reasoning is unclear.",
            "3": "Some actionable points, but rationale is incomplete or options are underspecified.",
            "5": "Actionable and easy to follow: clear rationale, clear options, and clear decision implications."
          }
        },
        "tradeoffs_risks": {
          "definition": "Communicates tradeoffs, risks, and assumptions clearly and concisely, matching the tone of a decision memo or case discussion.",
          "score_anchors": {
            "1": "One-sided or overconfident; ignores tradeoffs/risks; unclear assumptions.",
            "3": "Mentions tradeoffs/risks but generically or without linking them to the decision.",
            "5": "Explicit tradeoffs/risks/assumptions tied to the decision; concise and decision-ready framing."
          }
        }
      }
    },

    "engineering": {
      "dimensions": {
        "requirements_coverage": {
          "definition": "Captures the system/design problem, requirements, constraints, and the key components of the proposed solution.",
          "score_anchors": {
            "1": "Misses the core problem or omits key requirements/constraints/components.",
            "3": "Captures the main problem and solution idea, but misses important constraints or key components.",
            "5": "Clearly captures the problem, requirements, constraints, and the major system components from the slides."
          }
        },
        "technical_faithfulness": {
          "definition": "Represents technical details accurately (architecture, interfaces, parameters, performance claims) without inventing specs or behaviors.",
          "score_anchors": {
            "1": "Multiple incorrect technical claims or invented specs; contradicts slides.",
            "3": "Mostly accurate but with minor technical inaccuracies or unsupported extrapolations.",
            "5": "Technically accurate and grounded; no invented specs/behaviors; matches slide claims."
          }
        },
        "design_narrative": {
          "definition": "Presents a coherent engineering narrative (problem → requirements → design choices → implementation/architecture → evaluation).",
          "score_anchors": {
            "1": "Disorganized; mixes stages; no clear design narrative or rationale.",
            "3": "Some structure, but stages are blended or missing (e.g., design choices not connected to requirements).",
            "5": "Clear structure that ties requirements to design choices and follows through to implementation and evaluation."
          }
        },
        "system_explainability": {
          "definition": "Explains how the system works and why design choices were made, in a way a technical audience could follow at a high level.",
          "score_anchors": {
            "1": "Hard to understand; missing how/why; key mechanisms are unclear.",
            "3": "Understandable overall but missing important how/why details or leaving ambiguity in system behavior.",
            "5": "Clear explanation of how it works and why choices were made; easy to follow at a high level."
          }
        },
        "tradeoffs_limits": {
          "definition": "Communicates tradeoffs and evaluation criteria (performance, reliability, cost, safety) clearly, and notes limitations or failure modes when present.",
          "score_anchors": {
            "1": "Ignores tradeoffs/limits; overclaims performance or feasibility.",
            "3": "Mentions some tradeoffs/limits but not clearly tied to design/evaluation.",
            "5": "Explicit tradeoffs/criteria/limits tied to the design and evaluation; appropriately cautious where needed."
          }
        }
      }
    },

    "social_sciences": {
      "dimensions": {
        "research_question_constructs": {
          "definition": "Captures the core research question/claim, key constructs/variables, and the main relationships or hypotheses discussed.",
          "score_anchors": {
            "1": "Research question/claim is missing or wrong; key constructs are absent.",
            "3": "Main question and some constructs included, but important relationships/hypotheses are missing.",
            "5": "Clearly states the central question/claim and the key constructs and relationships emphasized on slides."
          }
        },
        "theory_fidelity": {
          "definition": "Represents theories, constructs, and conceptual frameworks accurately, using terms as the lecture uses them.",
          "score_anchors": {
            "1": "Misstates theories/constructs or invents frameworks; contradicts slides.",
            "3": "Mostly accurate but blurs key distinctions or misstates a concept/detail.",
            "5": "Accurate representation of theories/constructs/frameworks with correct distinctions."
          }
        },
        "method_evidence_alignment": {
          "definition": "Accurately describes methods/evidence (study design, measures, data) and how they relate to the claims.",
          "score_anchors": {
            "1": "Methods/evidence are wrong, missing, or disconnected from claims; invents study details.",
            "3": "Some method/evidence detail, but incomplete linkage to claims or missing key design/measure points.",
            "5": "Clear, slide-consistent method/evidence description and how it supports (or limits) the claims."
          }
        },
        "causal_nuance_confounding": {
          "definition": "Communicates nuance about causality vs correlation, confounding, and alternative explanations when relevant.",
          "score_anchors": {
            "1": "Overclaims causality or certainty; ignores confounding/alternatives.",
            "3": "Mentions some nuance but generically or without tying to specific claims.",
            "5": "Appropriately cautious and explicit about causality, confounding, and alternative explanations."
          }
        },
        "implications_generalizability": {
          "definition": "Summarizes implications and generalizability limits (who/where it applies) without overstating conclusions.",
          "score_anchors": {
            "1": "Invents implications or overgeneralizes beyond evidence; no limits mentioned.",
            "3": "Some implications/limits but vague or incomplete.",
            "5": "Slide-grounded implications with clear, appropriate limits on generalizability."
          }
        }
      }
    },

    "arts": {
      "dimensions": {
        "work_and_context": {
          "definition": "Identifies the central work(s)/practice and the relevant context (creator, period, movement, setting) emphasized in the slides.",
          "score_anchors": {
            "1": "Misses the central work(s) or provides incorrect/unsupported context.",
            "3": "Covers the main work(s) and some context, but misses important contextual elements.",
            "5": "Clearly captures the key work(s) and the most relevant context emphasized on slides."
          }
        },
        "formal_analysis": {
          "definition": "Describes formal elements (technique, composition, style, medium) concretely and accurately.",
          "score_anchors": {
            "1": "Formal elements are vague, incorrect, or largely missing.",
            "3": "Some concrete formal description, but incomplete or uneven accuracy.",
            "5": "Concrete, accurate formal analysis aligned with slide content."
          }
        },
        "interpretation_support": {
          "definition": "Connects interpretive claims to observable features, lecture points, or slide-grounded evidence; avoids invented details.",
          "score_anchors": {
            "1": "Interpretation is unsupported or invents details/claims not in slides.",
            "3": "Some support, but connections between features and meaning are weak or incomplete.",
            "5": "Interpretations are clearly supported by slide-grounded features and lecture framing."
          }
        },
        "comparisons_significance": {
          "definition": "Captures comparisons, influences, or significance claims (why it matters) when present on slides.",
          "score_anchors": {
            "1": "Misses major significance/comparison points emphasized on slides.",
            "3": "Mentions significance/comparisons but generically or incompletely.",
            "5": "Clearly captures slide-grounded significance and meaningful comparisons/influences."
          }
        },
        "nuance_multiple_readings": {
          "definition": "Acknowledges ambiguity or multiple readings when present and maintains an analysis tone without overclaiming.",
          "score_anchors": {
            "1": "Overconfident, purely opinion-based, or ignores slide-stated nuance.",
            "3": "Some nuance, but alternatives/ambiguity are mentioned weakly or without grounding.",
            "5": "Nuanced and grounded: acknowledges complexity and supports readings appropriately."
          }
        }
      }
    },

    "general": {
      "dimensions": {
        "coverage": {
          "definition": "Captures the main topics and key takeaways from the slides, prioritizing the most emphasized points over minor details.",
          "score_anchors": {
            "1": "Misses major points; summary is sparse or focused on minor details.",
            "3": "Covers main topics but misses some important takeaways or over-includes minor points.",
            "5": "Covers the key takeaways and the most important topics emphasized on slides."
          }
        },
        "faithfulness": {
          "definition": "Accurately reflects slide content without inventing facts, claims, or examples; uses correct terminology as presented.",
          "score_anchors": {
            "1": "Multiple inaccuracies or invented claims; contradicts slide content.",
            "3": "Mostly accurate with minor errors or occasional overreach beyond slides.",
            "5": "Accurate and slide-grounded; no invented claims or contradictions."
          }
        },
        "organization": {
          "definition": "Presents information in a logical order that mirrors the lecture flow or groups related points clearly.",
          "score_anchors": {
            "1": "Hard to follow; no clear structure or ordering.",
            "3": "Some structure, but ordering is uneven or grouping is inconsistent.",
            "5": "Clear organization: either follows lecture flow or uses strong thematic grouping."
          }
        },
        "clarity": {
          "definition": "Explains ideas in clear, concise language with minimal ambiguity; defines key terms when needed.",
          "score_anchors": {
            "1": "Confusing or overly vague; key terms are undefined or used inconsistently.",
            "3": "Generally understandable, but some parts are wordy, vague, or underexplained.",
            "5": "Clear and concise; key terms are defined or made understandable in context."
          }
        },
        "style": {
          "definition": "Maintains a readable, professional tone appropriate for lecture notes and avoids unnecessary verbosity or excessive informality.",
          "score_anchors": {
            "1": "Tone is inconsistent or distracting; overly verbose or overly casual.",
            "3": "Mostly appropriate tone, but could be tighter or more consistent.",
            "5": "Consistent, readable tone; succinct and appropriate for a lecture summary."
          }
        }
      }
    }
  }
}
