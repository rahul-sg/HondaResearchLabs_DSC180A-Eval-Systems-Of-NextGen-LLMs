{
  "domains": {
    "math": {
      "dimensions": {
        "coverage": {
          "definition": "Includes the key assumptions/conditions and scope under which the main results or methods apply.",
          "score_anchors": {
            "1": "Assumptions/conditions are missing or wrong; treats conditional results as universal.",
            "3": "Some assumptions/conditions included but important ones are missing or vague.",
            "5": "Key assumptions/conditions and applicability are clearly stated and match the slides."
          }
        },
        "faithfulness": {
          "definition": "Definitions, theorems, claims, and conclusions are stated correctly (including constraints) with no slide-unsupported inventions.",
          "score_anchors": {
            "1": "Major definitions/results are incorrect or fabricated; frequent unsupported claims.",
            "3": "Mostly correct but with missing conditions or minor inaccuracies; few unsupported claims.",
            "5": "Accurate statements of central definitions/results with essential conditions; no unsupported claims."
          }
        },
        "organization": {
          "definition": "Presents the reasoning flow (proof/derivation steps) in a coherent order that matches the lecture structure.",
          "score_anchors": {
            "1": "Disorganized or scrambled; reasoning chain is incoherent or contradicts slides.",
            "3": "Some logical flow, but gaps or mildly muddled sequencing of key steps.",
            "5": "Clear, slide-consistent structure: main result → key steps → conclusion."
          }
        },
        "clarity": {
          "definition": "Explains methods/procedures in a usable way (inputs → steps → outputs) and makes the takeaway easy to follow.",
          "score_anchors": {
            "1": "Procedure is unusable/unclear or incorrect; hard to follow.",
            "3": "Understandable but missing important steps/details or has ambiguity.",
            "5": "Procedure is clear, correct, and easy to apply based on the slides."
          }
        },
        "style": {
          "definition": "Uses precise notation/terminology and appropriately notes limitations/edge cases (without overclaiming).",
          "score_anchors": {
            "1": "Notation/terms are sloppy or wrong; overconfident claims with no limits.",
            "3": "Mostly readable with minor precision issues; mentions limits superficially or inconsistently.",
            "5": "Precise notation/terminology and appropriate limits/edge cases are clearly communicated."
          }
        }
      }
    },
    "humanities": {
      "dimensions": {
        "coverage": {
          "definition": "Captures the lecture’s central thesis/question and the main line of what it is arguing (not just a topic list).",
          "score_anchors": {
            "1": "No thesis or the thesis is wrong; mostly a list of topics.",
            "3": "Thesis is present but generic or partially misaligned; misses key parts of the main argument.",
            "5": "Thesis/question is explicit, accurate, and captures the core argumentative focus of the lecture."
          }
        },
        "faithfulness": {
          "definition": "Represents key concepts and distinctions accurately (terms used as the lecture uses them), without distortion or invention.",
          "score_anchors": {
            "1": "Key concepts are misdefined/flattened; major distinctions lost or reversed.",
            "3": "Most concepts are correct but some nuance/distinctions are blurred or underexplained.",
            "5": "Concepts and distinctions are accurate and preserve the lecture’s intended meanings."
          }
        },
        "organization": {
          "definition": "Shows how the lecture builds its argument (claim → reasoning → support) in a coherent progression.",
          "score_anchors": {
            "1": "Disjointed; no clear reasoning chain; argument flow is missing or fabricated.",
            "3": "Some structure, but linkage between claims and reasoning is incomplete or uneven.",
            "5": "Clear argumentative flow that matches how the lecture develops and supports its thesis."
          }
        },
        "clarity": {
          "definition": "Connects claims to lecture-discussed examples/texts/artifacts (as shown on slides) and avoids fabricated quotes or references.",
          "score_anchors": {
            "1": "Examples are missing or invented; fabricated quotes/references appear.",
            "3": "Some real examples included, but linkage is weak or incomplete; may be vague about support.",
            "5": "Examples are slide-grounded and clearly tied to the claims they support; no invention."
          }
        },
        "style": {
          "definition": "Communicates nuance: acknowledges tensions/counterpoints/ambiguities and explains stakes/implications emphasized in the lecture.",
          "score_anchors": {
            "1": "Overly certain, ignores debate, or invents implications not present in the lecture.",
            "3": "Mentions tension/stakes but generally or without clear connection to the lecture’s framing.",
            "5": "Accurately captures key tensions and meaningful implications grounded in the lecture."
          }
        }
      }
    },
    "natural_sciences": {
      "dimensions": {
        "coverage": {
          "definition": "Covers the core system/process and the key mechanisms (how/why) emphasized in the lecture.",
          "score_anchors": {
            "1": "Misses or misstates the main mechanism; major core content absent.",
            "3": "Includes the main mechanism but misses important steps/variables or is shallow.",
            "5": "Captures the main mechanism with the key steps/variables emphasized on slides."
          }
        },
        "faithfulness": {
          "definition": "Accurately represents models/diagrams/equations/frameworks and does not invent relationships not supported by slides.",
          "score_anchors": {
            "1": "Models/relationships are wrong or fabricated; contradicts slide representations.",
            "3": "Mostly correct but missing key structure or has minor misinterpretations.",
            "5": "Models/representations are accurately described with correct variable roles/relationships."
          }
        },
        "organization": {
          "definition": "Presents scientific logic in order (methods/evidence → results → conclusions) and makes the support for claims clear.",
          "score_anchors": {
            "1": "No valid evidence-to-claim chain; claims appear ungrounded or out of order.",
            "3": "Some linkage but incomplete experimental/observational logic or missing steps.",
            "5": "Clear, slide-consistent chain from evidence to results to conclusions."
          }
        },
        "clarity": {
          "definition": "Separates observed results from interpretation/hypotheses and avoids overstating certainty.",
          "score_anchors": {
            "1": "Conflates results with interpretation; misleading certainty or incorrect conclusions.",
            "3": "Some separation, but blurred in places or occasionally overstated.",
            "5": "Consistently distinguishes results vs interpretation and labels tentative claims appropriately."
          }
        },
        "style": {
          "definition": "Communicates uncertainty/limitations clearly and connects findings to implications, predictions, applications, or next questions (when present).",
          "score_anchors": {
            "1": "No limits/uncertainty; overconfident or invents implications/applications.",
            "3": "Mentions limits/implications but generic or without clear grounding.",
            "5": "Key uncertainty/limits and slide-grounded implications/next steps are clearly stated."
          }
        }
      }
    },
    "business": {
      "dimensions": {
        "coverage": {
          "definition": "Captures the decision context: the core problem, objective, and key situation facts the lecture/case is built around.",
          "score_anchors": {
            "1": "Problem/objective is missing or incorrect; key context is absent.",
            "3": "Problem/objective is stated but important context or constraints are missing.",
            "5": "Clearly states the core decision/problem, objectives, and the essential context from the slides."
          }
        },
        "faithfulness": {
          "definition": "Uses business terms, claims, and any numbers/metrics exactly as presented; does not invent data, outcomes, or relationships.",
          "score_anchors": {
            "1": "Invents metrics/results or misstates core facts; multiple unsupported claims.",
            "3": "Mostly accurate, but has minor factual/metric errors or some overreach beyond slides.",
            "5": "Accurate and slide-grounded; metrics and claims are represented correctly with no invented facts."
          }
        },
        "organization": {
          "definition": "Organizes the analysis in a decision-oriented flow (context → analysis → options/tradeoffs → recommendation).",
          "score_anchors": {
            "1": "Disorganized; lacks a coherent decision flow; options/recommendation are unclear or misplaced.",
            "3": "Some structure, but analysis/options/tradeoffs are not clearly separated or sequenced.",
            "5": "Clear progression from context to analysis to options/tradeoffs to a well-placed conclusion/recommendation."
          }
        },
        "clarity": {
          "definition": "Makes the reasoning actionable: clearly explains why certain conclusions follow and what actions/options are being considered.",
          "score_anchors": {
            "1": "Vague takeaways; hard to tell what to do or why; reasoning is unclear.",
            "3": "Some actionable points, but rationale is incomplete or options are underspecified.",
            "5": "Actionable and easy to follow: clear rationale, clear options, and clear decision implications."
          }
        },
        "style": {
          "definition": "Communicates tradeoffs, risks, and assumptions cleanly and concisely, matching the tone of a decision memo or case discussion.",
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
        "coverage": {
          "definition": "Covers the system/design problem, requirements, constraints, and the key components of the proposed solution.",
          "score_anchors": {
            "1": "Misses the core problem or omits key requirements/constraints/components.",
            "3": "Captures the main problem and solution idea, but misses important constraints or key components.",
            "5": "Clearly captures the problem, requirements, constraints, and the major system components from the slides."
          }
        },
        "faithfulness": {
          "definition": "Represents technical details accurately (architecture, interfaces, parameters, performance claims) without inventing specs or behaviors.",
          "score_anchors": {
            "1": "Multiple incorrect technical claims or invented specs; contradicts slides.",
            "3": "Mostly accurate but with minor technical inaccuracies or unsupported extrapolations.",
            "5": "Technically accurate and grounded; no invented specs/behaviors; matches slide claims."
          }
        },
        "organization": {
          "definition": "Presents a coherent engineering narrative (problem → requirements → design choices → implementation/architecture → evaluation).",
          "score_anchors": {
            "1": "Disorganized; mixes stages; no clear design narrative or rationale.",
            "3": "Some structure, but stages are blended or missing (e.g., design choices not connected to requirements).",
            "5": "Clear structure that ties requirements to design choices and follows through to implementation and evaluation."
          }
        },
        "clarity": {
          "definition": "Explains how the system works and why design choices were made, in a way that a technical audience could follow and reproduce at a high level.",
          "score_anchors": {
            "1": "Hard to understand; missing how/why; key mechanisms are unclear.",
            "3": "Understandable overall but missing important how/why details or leaving ambiguity in system behavior.",
            "5": "Clear explanation of how it works and why choices were made; easy to follow at a high level."
          }
        },
        "style": {
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
        "coverage": {
          "definition": "Captures the core research question or claim, the key variables/constructs, and the main relationships or hypotheses discussed.",
          "score_anchors": {
            "1": "Research question/claim is missing or wrong; key variables/constructs are absent.",
            "3": "Main question and some variables included, but important constructs or relationships are missing.",
            "5": "Clearly states the central question/claim and the key variables/constructs and relationships emphasized on slides."
          }
        },
        "faithfulness": {
          "definition": "Represents theories, constructs, study claims, and any reported findings accurately; avoids inventing causal claims or results not supported by slides.",
          "score_anchors": {
            "1": "Misstates theories/findings or invents results/causality; multiple unsupported claims.",
            "3": "Mostly accurate but occasionally overstates causality or misstates a construct/detail.",
            "5": "Accurate, slide-grounded representation of theories/constructs/findings; no invented results or causal leaps."
          }
        },
        "organization": {
          "definition": "Organizes content in a coherent social-science flow (question/theory → method/evidence → findings → interpretation/implications).",
          "score_anchors": {
            "1": "Disorganized; lacks a coherent chain from theory to evidence to conclusions.",
            "3": "Some structure, but method/evidence/findings/implications are blended or incompletely sequenced.",
            "5": "Clear progression from question/theory to evidence to findings to interpretation and implications."
          }
        },
        "clarity": {
          "definition": "Explains constructs in plain language, clarifies how they were measured or operationalized (when present), and makes claims easy to understand.",
          "score_anchors": {
            "1": "Constructs and claims are unclear; measurement/definitions are missing or confusing.",
            "3": "Mostly understandable, but definitions or operationalization details are incomplete or vague.",
            "5": "Clear definitions and explanations; measurement/operationalization and key claims are easy to follow."
          }
        },
        "style": {
          "definition": "Communicates nuance about causality, confounding, limitations, and generalizability; presents implications without overclaiming.",
          "score_anchors": {
            "1": "Overclaims causality or certainty; ignores limitations/confounding.",
            "3": "Mentions some limitations/nuance but generically or without tying to conclusions.",
            "5": "Appropriately cautious: clearly states limitations/nuance and aligns implication strength with the evidence."
          }
        }
      }
    },

    "arts": {
      "dimensions": {
        "coverage": {
          "definition": "Captures the central work(s) or practice discussed and the main formal elements (technique, composition, style, medium) and themes.",
          "score_anchors": {
            "1": "Misses the central work(s) or omits major formal elements/themes emphasized in the slides.",
            "3": "Covers the main work(s) and some elements/themes, but misses important aspects or context.",
            "5": "Clearly captures the key work(s), main formal elements, and central themes/context emphasized on slides."
          }
        },
        "faithfulness": {
          "definition": "Accurately describes the work(s), techniques, historical/contextual claims, and interpretations presented; avoids inventing details about the artwork or creator.",
          "score_anchors": {
            "1": "Invents details or misdescribes key attributes; contradicts slide content.",
            "3": "Mostly accurate but with minor descriptive errors or interpretive claims not clearly grounded.",
            "5": "Accurate and grounded in slide content; no invented details or misattributions."
          }
        },
        "organization": {
          "definition": "Organizes the discussion coherently (context → formal analysis → interpretation → significance/comparison).",
          "score_anchors": {
            "1": "Disorganized; formal analysis and interpretation are mixed without clear flow.",
            "3": "Some flow, but sections are blended or missing (e.g., context not connected to interpretation).",
            "5": "Clear structure connecting context, formal elements, interpretation, and significance."
          }
        },
        "clarity": {
          "definition": "Uses accessible, concrete language to describe formal elements and links them explicitly to meaning/interpretation.",
          "score_anchors": {
            "1": "Vague or purely subjective; unclear how formal elements connect to interpretation.",
            "3": "Some clear description, but connections between form and meaning are incomplete or inconsistent.",
            "5": "Concrete descriptions with clear form-to-meaning connections; easy for a listener to follow."
          }
        },
        "style": {
          "definition": "Balances description and interpretation, acknowledges multiple readings when present, and matches the tone of critique/analysis without overclaiming.",
          "score_anchors": {
            "1": "Overconfident or purely opinion-based; ignores alternative readings or slide-stated nuance.",
            "3": "Some nuance, but interpretation may dominate without grounding or alternatives are mentioned weakly.",
            "5": "Nuanced and grounded: appropriate balance of description and interpretation with acknowledged complexity."
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


# Python helper functions
import json


def get_all_domains() -> list[str]:
  """Return all supported schema domains for domain-aware evaluation."""
  return [
    "math",
    "humanities",
    "natural_sciences",
    "engineering",
    "social_sciences",
    "business",
  ]

def get_domain_schema(domain: str) -> dict:
    """Return the evaluation schema for a specific academic domain.

    Currently the schemas are hard‑coded in this function. In future we
    could load them from an external JSON file if desired.
    """

    # The full schema is defined below; keep in sync with the JSON-like
    # structure present earlier in this module.  This replaces the previous
    # attempt to load `__file__` as JSON, which was causing a decode error.
    full_schema = {
      "domains": {
        "math": {
          "dimensions": {
            "coverage": {
              "definition": "Includes the key assumptions/conditions and scope under which the main results or methods apply.",
              "score_anchors": {
                "1": "Assumptions/conditions are missing or wrong; treats conditional results as universal.",
                "3": "Some assumptions/conditions included but important ones are missing or vague.",
                "5": "Key assumptions/conditions and applicability are clearly stated and match the slides."
              }
            },
            "faithfulness": {
              "definition": "Definitions, theorems, claims, and conclusions are stated correctly (including constraints) with no slide-unsupported inventions.",
              "score_anchors": {
                "1": "Major definitions/results are incorrect or fabricated; frequent unsupported claims.",
                "3": "Mostly correct but with missing conditions or minor inaccuracies; few unsupported claims.",
                "5": "Accurate statements of central definitions/results with essential conditions; no unsupported claims."
              }
            },
            "organization": {
              "definition": "Presents the reasoning flow (proof/derivation steps) in a coherent order that matches the lecture structure.",
              "score_anchors": {
                "1": "Disorganized or scrambled; reasoning chain is incoherent or contradicts slides.",
                "3": "Some logical flow, but gaps or mildly muddled sequencing of key steps.",
                "5": "Clear, slide-consistent structure: main result → key steps → conclusion."
              }
            },
            "clarity": {
              "definition": "Explains methods/procedures in a usable way (inputs → steps → outputs) and makes the takeaway easy to follow.",
              "score_anchors": {
                "1": "Procedure is unusable/unclear or incorrect; hard to follow.",
                "3": "Understandable but missing important steps/details or has ambiguity.",
                "5": "Procedure is clear, correct, and easy to apply based on the slides."
              }
            },
            "style": {
              "definition": "Uses precise notation/terminology and appropriately notes limitations/edge cases (without overclaiming).",
              "score_anchors": {
                "1": "Notation/terms are sloppy or wrong; overconfident claims with no limits.",
                "3": "Mostly readable with minor precision issues; mentions limits superficially or inconsistently.",
                "5": "Precise notation/terminology and appropriate limits/edge cases are clearly communicated."
              }
            }
          }
        },
        "humanities": {
          "dimensions": {
            "coverage": {
              "definition": "Captures the lecture's central thesis/question and the main line of what it is arguing (not just a topic list).",
              "score_anchors": {
                "1": "No thesis or the thesis is wrong; mostly a list of topics.",
                "3": "Thesis is present but generic or partially misaligned; misses key parts of the main argument.",
                "5": "Thesis/question is explicit, accurate, and captures the core argumentative focus of the lecture."
              }
            },
            "faithfulness": {
              "definition": "Represents key concepts and distinctions accurately (terms used as the lecture uses them), without distortion or invention.",
              "score_anchors": {
                "1": "Key concepts are misdefined/flattened; major distinctions lost or reversed.",
                "3": "Most concepts are correct but some nuance/distinctions are blurred or underexplained.",
                "5": "Concepts and distinctions are accurate and preserve the lecture's intended meanings."
              }
            },
            "organization": {
              "definition": "Shows how the lecture builds its argument (claim → reasoning → support) in a coherent progression.",
              "score_anchors": {
                "1": "Disjointed; no clear reasoning chain; argument flow is missing or fabricated.",
                "3": "Some structure, but linkage between claims and reasoning is incomplete or uneven.",
                "5": "Clear argumentative flow that matches how the lecture develops and supports its thesis."
              }
            },
            "clarity": {
              "definition": "Connects claims to lecture-discussed examples/texts/artifacts (as shown on slides) and avoids fabricated quotes or references.",
              "score_anchors": {
                "1": "Examples are missing or invented; fabricated quotes/references appear.",
                "3": "Some real examples included, but linkage is weak or incomplete; may be vague about support.",
                "5": "Examples are slide-grounded and clearly tied to the claims they support; no invention."
              }
            },
            "style": {
              "definition": "Communicates nuance: acknowledges tensions/counterpoints/ambiguities and explains stakes/implications emphasized in the lecture.",
              "score_anchors": {
                "1": "Overly certain, ignores debate, or invents implications not present in the lecture.",
                "3": "Mentions tension/stakes but generally or without clear connection to the lecture's framing.",
                "5": "Accurately captures key tensions and meaningful implications grounded in the lecture."
              }
            }
          }
        },
        "natural_sciences": {
          "dimensions": {
            "coverage": {
              "definition": "Covers the core system/process and the key mechanisms (how/why) emphasized in the lecture.",
              "score_anchors": {
                "1": "Misses or misstates the main mechanism; major core content absent.",
                "3": "Includes the main mechanism but misses important steps/variables or is shallow.",
                "5": "Captures the main mechanism with the key steps/variables emphasized on slides."
              }
            },
            "faithfulness": {
              "definition": "Accurately represents the scientific content, relationships, and evidence without distortion or fabrication.",
              "score_anchors": {
                "1": "Major scientific inaccuracies or fabrications; relationships between concepts are wrong.",
                "3": "Mostly accurate but with some distortions or missing relationships between key concepts.",
                "5": "Accurate representation of scientific content and relationships as presented in the lecture."
              }
            },
            "organization": {
              "definition": "Presents the scientific reasoning (hypothesis → evidence → conclusion) in a logical sequence that matches the lecture.",
              "score_anchors": {
                "1": "Disorganized; scientific reasoning chain is incoherent or contradicts the lecture.",
                "3": "Some logical flow, but gaps in the scientific reasoning or sequencing issues.",
                "5": "Clear scientific reasoning flow that matches the lecture's hypothesis-evidence-conclusion structure."
              }
            },
            "clarity": {
              "definition": "Explains scientific concepts/procedures with clear connections to evidence and avoids oversimplification or confusion.",
              "score_anchors": {
                "1": "Scientific concepts are oversimplified to the point of inaccuracy or are confusing.",
                "3": "Understandable but some concepts are oversimplified or connections to evidence are unclear.",
                "5": "Clear explanation of scientific concepts with appropriate complexity and strong evidence connections."
              }
            },
            "style": {
              "definition": "Uses appropriate scientific terminology and acknowledges uncertainties/limitations of the science discussed.",
              "score_anchors": {
                "1": "Scientific terminology misused; ignores uncertainties or presents science as absolute fact.",
                "3": "Mostly appropriate terminology but may overstate certainty or miss some limitations.",
                "5": "Appropriate scientific terminology and balanced discussion of uncertainties/limitations."
              }
            }
          }
        },
        "business": {
          "dimensions": {
            "coverage": {
              "definition": "Covers the core business concepts, frameworks, and practical applications emphasized in the lecture.",
              "score_anchors": {
                "1": "Misses core business concepts/frameworks; major content absent or wrong.",
                "3": "Includes main concepts but misses important frameworks/applications or is superficial.",
                "5": "Captures core business concepts, frameworks, and applications as emphasized on slides."
              }
            },
            "faithfulness": {
              "definition": "Accurately represents business concepts, relationships, and implications without distortion or fabrication.",
              "score_anchors": {
                "1": "Major business concept inaccuracies or fabrications; relationships between concepts are wrong.",
                "3": "Mostly accurate but with some distortions or missing relationships between key business concepts.",
                "5": "Accurate representation of business concepts and relationships as presented in the lecture."
              }
            },
            "organization": {
              "definition": "Presents business reasoning (problem → analysis → solution/recommendation) in a logical sequence that matches the lecture.",
              "score_anchors": {
                "1": "Disorganized; business reasoning chain is incoherent or contradicts the lecture.",
                "3": "Some logical flow, but gaps in business reasoning or sequencing issues.",
                "5": "Clear business reasoning flow that matches the lecture's problem-analysis-solution structure."
              }
            },
            "clarity": {
              "definition": "Explains business concepts/applications with clear connections to real-world implications and avoids oversimplification.",
              "score_anchors": {
                "1": "Business concepts oversimplified to inaccuracy or confusing; no real-world connections.",
                "3": "Understandable but some concepts oversimplified or real-world implications unclear.",
                "5": "Clear business explanations with appropriate complexity and strong real-world connections."
              }
            },
            "style": {
              "definition": "Uses appropriate business terminology and acknowledges practical constraints/limitations of business strategies.",
              "score_anchors": {
                "1": "Business terminology misused; ignores practical constraints or presents strategies as universally applicable.",
                "3": "Mostly appropriate terminology but may overstate effectiveness or miss practical limitations.",
                "5": "Appropriate business terminology and realistic discussion of constraints/limitations."
              }
            }
          }
        }
      }
    }

    return full_schema["domains"].get(domain, full_schema["domains"]["humanities"])
