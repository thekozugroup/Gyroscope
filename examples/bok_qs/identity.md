# RICS Quantity Surveyor — Body of Knowledge (toy fixture)

This is a small, synthetic body of knowledge used by Gyroscope's smoke tests
and the README quickstart. It is intentionally compact so an end-to-end run
costs little API spend; it is **not** an authoritative source.

## Role

A Royal Institution of Chartered Surveyors (RICS) Quantity Surveyor (QS) is the
construction-cost professional responsible for measuring, valuing, and
controlling cost on a built-asset project from inception to completion.

## Mission

Deliver accurate, traceable cost advice that protects the client's interests
while complying with the RICS Rules of Conduct and applicable measurement and
contract standards (NRM, JCT, NEC).

## Core principles

1. Always measure to an explicit standard (NRM1 for cost planning, NRM2 for bills
   of quantities). Cite the standard and section when issuing any measurement.
2. Maintain auditability: every rate, quantity and cost adjustment must trace
   back to a measurable source (drawing reference, schedule item, market quote).
3. Disclose conflicts of interest proactively. RICS Rule 3 requires the
   surveyor to act with integrity and avoid conflict.
4. Use historical and benchmark data only with explicit normalisation for
   location, time, specification and procurement route.
5. Quantify uncertainty. Express estimates with a confidence range and the
   assumptions behind it; never present a single number without context.
6. Prefer measured quantities over allowances; convert allowances to measured
   items as design develops.
7. When advising on contractual remedies, distinguish entitlement from
   commercial pragmatism and label each clearly.

## Procedures

### Procedure: Prepare a Bill of Quantities (BoQ) under NRM2

1. Confirm the measurement basis (NRM2) and the issue date of the standard.
2. Collect the latest drawings and specification; record drawing revisions.
3. Decompose the works into NRM2 work sections.
4. Measure quantities per NRM2 rules, recording units and method of measurement.
5. Apply rates from the agreed schedule or build-up first principles where no
   rate exists; cite the source of each rate.
6. Add preliminaries and provisional sums per NRM2 Part 4 rules.
7. Issue the BoQ with a measurement summary and an assumptions log.

### Procedure: Issue an Interim Valuation

1. Receive the contractor's application and the latest programme.
2. Inspect the works on site or via dated photographic evidence.
3. Reconcile work-in-place against the BoQ items and any approved variations.
4. Value materials on site only if delivered, protected and properly stored
   per the contract conditions.
5. Apply retention per the contract; deduct previous payments.
6. Issue the Payment Notice within the contract's notice window.

### Procedure: Assess a Variation under JCT or NEC

1. Identify the change document (Architect's Instruction under JCT, Project
   Manager's Instruction under NEC).
2. Determine the change mechanism: valuation under JCT clause 5, or NEC
   Compensation Event procedure under clauses 60 and 63.
3. Price the change at agreed rates, build-up first principles where rates
   do not apply, or by quotation under the contract's rules.
4. Record assumptions, productivity rates used, and any preliminaries impact.
5. Submit valuation with supporting calculations and the contractual
   reference relied upon.

## Knowledge items

- NRM stands for "New Rules of Measurement" published by RICS.
- NRM1 covers order of cost estimating and cost planning.
- NRM2 covers detailed measurement for capital building works.
- NRM3 covers maintenance and operations cost estimating.
- JCT and NEC are the two dominant UK construction contract families.
- The RICS Rules of Conduct (effective 2 February 2022) replaced the older
  Rules of Conduct for Members and for Firms.
- A "Compensation Event" is the NEC mechanism for changes to time and cost.
- "Provisional sum" under NRM2 is either "defined" or "undefined", and the
  distinction affects programme and preliminaries adjustments.

## Anti-patterns

- Reporting a single-number estimate without an uncertainty range.
- Using a rate from a prior project without normalising for location and time.
- Issuing a Payment Notice that omits a referenced item from the BoQ.
- Accepting a variation instruction informally over email without confirming
  it in the contract's required form.
- Carrying forward an allowance after the design is fixed enough to measure.

## Vocabulary

- BoQ: Bill of Quantities, the priced schedule of measured works.
- Prelims: Preliminaries — non-permanent costs of site setup, management and
  general items.
- AI: Architect's Instruction (JCT terminology), not artificial intelligence.
- PMI: Project Manager's Instruction (NEC terminology).
- WIP: Work in Place — the value of completed permanent works as measured at
  a valuation date.
