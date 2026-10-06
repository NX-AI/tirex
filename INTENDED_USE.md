# TiRex Intended Use, Limitations and EU AI Act Notice

- **Notice version:** 1.0
- **Date:** 22 September 2026
- **Applies to:** the TiRex repository, model releases identified in that repository, and associated inference software
- **Licence:** NXAI Community Licence

## 1. Purpose and status of this notice

This notice describes NXAI's intended purpose for TiRex, material model and deployment limitations, and expectations for downstream integration. It is product and regulatory information, not legal advice.

This notice does not amend, replace, or restrict the NXAI Community Licence. Licence permissions, conditions, attribution requirements, and commercial-use restrictions are governed exclusively by the applicable licence text. Because that licence contains commercial restrictions, this notice does not characterise TiRex as free and open-source software for the purpose of any regulatory exemption.

Each actor remains responsible for determining and fulfilling the legal obligations applicable to its own role and use case.

## 2. Intended purpose

TiRex comprises xLSTM-based models and software for documented time-series tasks. The principal public use is zero-shot time-series forecasting with point and quantile predictions. The repository also documents separate classification and regression variants. Users must identify the exact model, task, and release they deploy and follow the corresponding model documentation.

TiRex is intended for research, evaluation, and non-high-risk decision support. It is not intended to make autonomous decisions, operate machinery, or replace qualified human judgement.

For forecasting and predictive-maintenance use, TiRex is intended to support analysis and planning. It is not intended to act as a safety component, trigger an automatic shutdown, or provide the sole basis for a maintenance or safety decision.

## 3. Excluded intended uses

NXAI does not intend TiRex to be used for any prohibited AI practice under Article 5 of the EU AI Act.

NXAI clearly specifies that TiRex is not to be changed into, integrated into, or materially relied upon as a high-risk AI system within the meaning of Article 6 and Annexes I and III of the EU AI Act. This includes use as a safety component of a product covered by Annex I and use as the sole or determinative basis for decisions in high-risk areas such as critical infrastructure, education, employment, essential services, law enforcement, migration, or the administration of justice.

Whether a particular downstream application is high-risk depends on its intended purpose, functionality, and deployment context. Use in a sector mentioned in the AI Act is not, by itself, a complete classification.

This section defines NXAI's intended purpose. It is not a licence restriction and does not purport to transfer, exclude, or reallocate obligations imposed by law.

## 4. Model limitations

- Forecast, classification, or regression quality depends on the relevance, quality, frequency, length, preprocessing, and representativeness of the input data.
- Performance reported on research benchmarks does not establish performance, safety, or regulatory compliance for a specific operational deployment.
- Distribution shift, rare events, regime changes, missing or erroneous observations, and unsuitable preprocessing can materially reduce model quality.
- Quantile forecasts express model-estimated uncertainty. They are not a guarantee of real-world coverage or application-level calibration.
- Classification and regression variants require task-specific validation. Integrators must assess label quality, class imbalance, decision thresholds, calibration, and the consequences of false positive and false negative results.
- TiRex does not provide causal explanations or a built-in explanation of individual outputs. External techniques may be evaluated by the integrator, but their suitability and correctness are not guaranteed by NXAI.
- The model does not provide application-level human oversight, access control, audit logging, fail-safe behaviour, or machinery control logic. These are integration responsibilities.

## 5. Deployment and data responsibilities

NXAI does not operate the generally available TiRex release as a hosted model service and therefore does not receive or retain deployment inputs, outputs, or inference logs. The deployer is responsible for appropriate logging, retention, access control, security, data protection, and incident handling. Customer-specific development or training engagements are governed by their applicable contracts.

Integrators must review the repository documentation and the NXAI Community Licence for release-specific deployment and commercial-use conditions.

## 6. Integration and validation responsibilities

Before operational use, the integrating organisation should:

- define the intended purpose, users, operating environment, and foreseeable misuse of the resulting AI system;
- identify the exact TiRex model, release, task, configuration, and licence terms;
- determine the organisation's role and the regulatory classification of the resulting AI system;
- validate task performance, robustness, uncertainty or score calibration, latency, resource requirements, and failure behaviour on representative data;
- establish human review, override, escalation, and safe fallback procedures proportionate to the consequences of an incorrect output;
- prevent model outputs from directly controlling safety-relevant machinery without independently validated safeguards;
- implement suitable versioning, monitoring, and logging without recording personal, confidential, or security-sensitive data unnecessarily;
- review data rights, data protection, cybersecurity, and sector-specific requirements; and
- monitor performance and reassess the deployment when data, operating conditions, model versions, or intended purpose change.

## 7. Responsibilities under the EU AI Act

This notice does not constitute regulatory approval, a conformity assessment, or a binding classification of TiRex or any downstream AI system.

Under Article 25 of the EU AI Act, a downstream actor may assume provider obligations when, for example, it places a high-risk AI system on the market under its own name, substantially modifies such a system, or changes its intended purpose so that it becomes high-risk. The application of Article 25 must be assessed for the specific model, actor, and deployment.

Each actor remains responsible for obligations applicable to its own role. NXAI's stated intended purpose does not remove obligations that the law assigns to NXAI, an integrator, a deployer, or another actor.

## 8. Further information

- Repository: <https://github.com/NX-AI/tirex>
- Documentation: <https://nx-ai.github.io/tirex/>
- NXAI Community Licence: <https://github.com/NX-AI/tirex/blob/main/LICENSE>
- EU AI Act, consolidated text: <https://eur-lex.europa.eu/eli/reg/2024/1689/2026-07-27/eng>
- Commission guidelines on general-purpose AI models: <https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-3>

- **Provider contact:** NXAI GmbH, Peter-Behrens-Platz 2, 4020 Linz, Austria
- **Company register:** FN 616894 y, Landesgericht Linz
- **VAT ID:** ATU80117419
- **Website:** <https://www.nx-ai.com>
- **Contact:** <contact@nx-ai.com>
