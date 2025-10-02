# ChatGPT_USAGE.md
Project Charter: Effective Use of ChatGPT Across Domains

This document defines protocols, invariants, skeleton templates, and checkpoints for structured collaboration with ChatGPT.  
It ensures reproducibility, clarity, and efficiency across coding, travel, fantasy football, recipes, and other applications.

------------------------------------------------------------
§A — General Protocols
------------------------------------------------------------
1. Atomic Requests
   - One request = one deliverable.
   - Multi-step tasks must be explicitly staged (Step 1, Step 2…).

2. Information Flow
   - Chunk information; provide only what is needed for the current step.
   - Use bookmarks to reference frozen context.

3. Checkpoints / Bookmarks
   ## BOOKMARK: <label>
   RESUME BOOKMARK: <label>
   SWITCH PROJECT: <label>

4. Error Handling
   - Fail fast with error codes:
     - USAGE:Overload → too much info in one request.
     - USAGE:UnknownProtocol → request does not follow domain protocol.
     - USAGE:Ambiguity → unclear request.

5. Commit Discipline
   - One logical change = one commit.
   - Commits must be staged and described before moving on to the next coding step.
   - Assistant must always prompt: **"Stage + commit now?"**
     - If user confirms, provide exact `git add … && git commit -m …` lines with a suggested descriptive message.
   - Commits should be atomic, descriptive, and pushed daily.
   - Scope: applies globally (MATLAB, Python, configs, docs).
   - Especially enforced for new files, completed features, and bug fixes.

------------------------------------------------------------
§B — Domain Protocols + Skeletons
------------------------------------------------------------

B1 — Coding
Protocol: Code Injection Protocol (CIP).
- File name
- Insertion point (with nearby lines)
- Full paste-ready snippet (no ellipses)
- Classification: Driver | Helper | Plotting

Skeleton:
## CODE REQUEST
File: <filename>.m
Classification: <Driver | Helper | Plotting>
Insertion Point:
    (unique nearby lines)
---
<full paste-ready snippet here>

Error codes:
- USAGE:CIPMissing → CIP not followed
- USAGE:InvariantBroken → rules violated

------------------------------------------------------------

B2 — Travel
Protocol:
- Trip name + date range
- Alliance constraint (Star Alliance required | Flexible)
- Deliverables: flights, cost breakdown, optional chart/TL;DR

Skeleton:
## TRAVEL REQUEST
Trip: <Trip name>
Dates: <YYYY-MM-DD> → <YYYY-MM-DD>
Alliance: <Star Alliance required | Flexible>
Deliverable:
    - Exact flight list (copy-paste ready)
    - Cost breakdown (cash, points, vouchers)
    - [Optional] Pie chart / TL;DR summary

Invariants:
- Must state alliance compliance.
- TL;DR summary available for non-technical audience.

------------------------------------------------------------

B3 — Fantasy Football
Protocol:
- Team name + roster snapshot
- Proposal: trade or waiver idea
- Deliverables: fairness check, source labels, paste-ready text

Skeleton:
## FANTASY REQUEST
Team: <Team name>
Roster Snapshot:
    QB: ...
    RB: ...
    WR: ...
    TE: ...
    FLEX: ...
    Bench: ...
    IR: ...
Proposal:
    - Trade or Waiver idea
Deliverable:
    - Fairness check
    - Web vs Memory sources explicitly labeled
    - Paste-ready trade text

Error codes:
- USAGE:RosterMismatch → roster conflicts with stored memory

------------------------------------------------------------

B4 — Recipes
Protocol:
- Dish name
- Ingredients + steps
- Variations clearly labeled
- Format: Cookbook | PDF style | Family-friendly | Vulgar humor

Skeleton:
## RECIPE REQUEST
Dish: <Name>
Ingredients:
    - item, qty
    - item, qty
Steps:
    1. ...
    2. ...
Variations:
    - Spicy: ...
    - Vegan: ...
Format:
    <Cookbook | PDF style | Family-friendly | Vulgar humor>

Invariants:
- Substitutions must be explicit.
- Times and temps must be consistent.

------------------------------------------------------------
B5 — Python + GitHub
------------------------------------------------------------
Protocol:
- Project name (and repo name/path if relevant)
- File name (.py, .yml, etc.)
- Insertion point (with nearby lines, or “top of file” / “end of file”)
- Clear task description
- Deliverables must be paste-ready code (no ellipses)
- Classification: Script | Helper | Test | Config
- If relevant, GitHub workflow alignment (CI, actions, tests)

Skeleton:
## PYTHON+GITHUB REQUEST
Project: <project name>
Repo: <repo-name or path>
File: <filename>.py
Insertion Point:
    (unique nearby lines, or "top of file" / "end of file")
Task:
    <clear description of what you want done>
Deliverable:
    - Paste-ready code snippet (no ellipses)
    - Classification: <Script | Helper | Test | Config>
    - GitHub workflow alignment if relevant

Error codes:
- USAGE:Ellipses → response included incomplete code
- USAGE:FileMismatch → wrong file or repo
- USAGE:ProtocolDrift → not following Python+GitHub skeleton

------------------------------------------------------------
§C — Memory & Resume
------------------------------------------------------------
- Project contexts persist under labels (e.g., AO Photometry, Fantasy, MTGA).
- Use BOOKMARK/RESUME/SWITCH to control context.

------------------------------------------------------------
§D — Extensions
------------------------------------------------------------
- New domains can be added with Protocol → Skeleton → Invariants → Error codes.
- Updates must be checkpointed with date.