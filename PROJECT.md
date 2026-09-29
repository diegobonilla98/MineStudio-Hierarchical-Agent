# PROJECT.md

# Hierarchical LLM + RL Agent for Minecraft

## 1. Project Summary

This project builds a general-purpose Minecraft agent using a hierarchical architecture that separates:

1. **High-level reasoning and planning**
2. **Grounded visuomotor control**
3. **Deterministic task verification**

The central idea is:

> Do not make reinforcement learning solve Minecraft end-to-end. Make reinforcement learning solve a distribution of grounded, verifiable, temporally local Minecraft control problems, while a multimodal LLM converts long-horizon objectives into those problems.

The architecture is intentionally hybrid:

- A multimodal LLM handles semantic reasoning, world understanding, planning, replanning, tool use, and long-horizon decomposition.
- A goal-conditioned learned policy handles visual interaction, movement, camera control, mining, combat, placement, and recovery from local mistakes.
- A deterministic verifier inspects environment state and decides whether goals have succeeded, failed, or progressed.
- Human demonstrations provide a strong behavior prior and prevent cold-start exploration.
- Online RL improves beyond demonstrations while remaining close to competent pretrained behavior.
- Optional closed-loop action primitives handle trivial repeated mechanics such as holding attack until a block breaks.

The first implementation should use **MineStudio** as the primary environment.

The initial executor should be initialized from a **Minecraft-pretrained VPT-style policy**, then trained using:

1. behavior cloning,
2. hindsight goal relabeling,
3. human interventions / DAgger-style corrections,
4. procedural task curriculum,
5. KL-regularized PPO.

MPC is intentionally excluded from V1.

---

# 2. Research Hypothesis

Minecraft is difficult for standard reinforcement learning because it combines:

- partial observability,
- extremely large action and state spaces,
- long-horizon dependencies,
- sparse rewards,
- semantic reasoning,
- exploration,
- visual grounding,
- hierarchical planning,
- inventory/resource management,
- local motor control,
- recovery from mistakes.

End-to-end RL asks one model to solve all of these simultaneously.

This project decomposes the problem according to the strengths of different systems.

## Multimodal LLM

Good at:

- semantic reasoning,
- task decomposition,
- recipe reasoning,
- selecting useful intermediate objectives,
- understanding inventory,
- interpreting visual scenes,
- maintaining long-term intent,
- replanning,
- selecting tools,
- reasoning about failures,
- forming structured objectives.

Bad at:

- 20 Hz motor control,
- precise camera motion,
- visual servoing,
- continuous local correction,
- low-latency interaction.

## Learned visuomotor policy

Good at:

- movement,
- camera control,
- combat,
- mining,
- placement,
- visual navigation,
- handling small disturbances,
- learning reactive behavior,
- improving through imitation and RL.

Bad at:

- very long credit assignment,
- semantic task decomposition,
- large combinatorial planning spaces,
- deciding why a task matters.

## Deterministic verifier

Good at:

- exact success detection,
- exact failure detection,
- environment-grounded reward,
- automatically labeling trajectories,
- preventing reward-model ambiguity.

The resulting architecture is:

```text
LONG-HORIZON OBJECTIVE
        |
        v
MULTIMODAL LLM
        |
        | structured GoalPacket
        v
GOAL-CONDITIONED EXECUTOR
        |
        | raw controls / action options
        v
MINECRAFT
        |
        +------------------+
        |                  |
        v                  v
OBSERVATIONS          VERIFIER
        |                  |
        +--------+---------+
                 |
        progress / success /
          failure / events
                 |
                 v
        EXECUTOR OR LLM
```

---

# 3. Primary Environment

## 3.1 MineStudio

Use **MineStudio** as the primary V1 environment.

Reasons:

- playable Minecraft GUI,
- human keyboard/mouse control,
- ability to switch between human and agent control,
- trajectory recording,
- image observations,
- structured environment information,
- custom callbacks,
- Minecraft command execution,
- inventory/state manipulation,
- custom rewards,
- offline datasets,
- online RL support,
- pretrained Minecraft policies,
- VPT-compatible action spaces,
- distributed execution with Ray,
- existing Minecraft research ecosystem.

MineStudio is particularly valuable because the same environment can support:

- human demonstration collection,
- autonomous agent execution,
- task generation,
- evaluation,
- RL training,
- policy debugging.

## 3.2 Future alternative environments

The task, verifier, and dataset APIs must remain environment-independent.

Possible future backends:

### CraftGround

Useful when:

- cleaner simulator instrumentation is required,
- modern Minecraft versions matter,
- structured observations are especially important,
- high simulation throughput is required,
- a custom policy stack is preferred.

### Bedrock-RL

Useful when:

- thousands of deterministic parallel RL environments are required,
- reproducibility across seeds is critical,
- large-scale automated curriculum training becomes dominant,
- exact synthetic experts or task-generation infrastructure is useful.

### MineRL

Do not use raw MineRL as the main V1 interface unless a specific compatibility requirement appears.

MineStudio provides a more complete research workflow.

---

# 4. Recommended Full Stack

## Environment

```text
MineStudio
Minecraft Java Edition backend
Linux / Ubuntu
```

Avoid making WSL the primary research environment.

## Core ML

```text
Python 3.10+
PyTorch
Transformers
einops
numpy
opencv-python
Pydantic
```

## Distributed simulation

```text
Ray
```

## Dataset

Primary trajectory storage:

```text
MineStudio trajectory format / LMDB
```

Metadata and experiment indexing:

```text
Parquet
DuckDB or Polars
```

Optional larger-scale storage:

```text
WebDataset
```

## Experiment tracking

```text
Weights & Biases
```

## Planner API

Use a provider-independent abstraction.

The planner may use:

- GPT-class multimodal models,
- Gemini-class multimodal models,
- another video-capable or image-capable multimodal LLM.

The code must not depend on one vendor.

## Validation and typed interfaces

```text
Pydantic
```

All planner output should be validated against strict schemas.

## Optional perception

Possible future modules:

```text
SAM-style segmentation
DINO-family dense features
object detection
depth estimation
tracking
```

These are auxiliary modules, not required in V1.

---

# 5. Core System Components

The system should be divided into the following modules.

```text
Planner
World Memory
Goal Compiler
Verifier
Perception / Grounding
Executor Policy
Action Option Library
Environment Adapter
Human Demonstration Recorder
Trajectory Dataset
Training Pipeline
Curriculum Scheduler
Evaluation Harness
```

Each component must have a clean interface.

---

# 6. Information Privilege Model

This distinction is mandatory.

There are four levels of information.

```text
1. Environment truth
2. Verifier truth
3. Training-only privileged information
4. Agent-visible observation
```

## Environment truth

May include:

- exact block map,
- hidden entities,
- underground resources,
- internal Minecraft state,
- exact world coordinates,
- arbitrary debug information.

## Verifier

May use privileged environment truth to evaluate whether a task succeeded.

Example:

```text
InventoryAtLeast("minecraft:diamond", 1)
```

The verifier may know exact inventory state even if the policy is trained from pixels.

## Training critic / teacher

May optionally use privileged information.

Examples:

- exact coordinates,
- target object ID,
- segmentation mask,
- distance-to-goal,
- oracle progress.

This enables asymmetric actor-critic training.

## Agent

Must only receive information allowed by the experiment.

Never accidentally expose privileged state to the actor.

Example failure:

The planner should not tell the policy that iron exists behind a wall unless that information was legitimately observed or retrieved from allowed memory.

---

# 7. Goal Representation

Do not use free-form language as the only interface between LLM and executor.

The LLM must produce a typed `GoalPacket`.

Example:

```json
{
  "goal_id": "mine_visible_stone_013",
  "parent_goal_id": "obtain_cobblestone",
  "skill": "mine_block",
  "instruction": "Mine the visible stone block ahead.",
  "target": {
    "entity_type": "minecraft:stone",
    "image_point": [0.61, 0.48],
    "mask_id": "mask_003"
  },
  "success": {
    "predicate": "inventory_delta",
    "item": "minecraft:cobblestone",
    "minimum": 1
  },
  "failure": [
    {
      "predicate": "health_below",
      "value": 4
    },
    {
      "predicate": "target_unreachable"
    }
  ],
  "max_steps": 240,
  "allowed_action_modes": [
    "raw",
    "attack_until_event",
    "select_hotbar"
  ]
}
```

A goal can contain:

- semantic skill category,
- natural-language instruction,
- object/entity identity,
- image point,
- image mask,
- reference crop,
- coordinates if legally available,
- preconditions,
- success predicates,
- failure predicates,
- timeout,
- maximum action horizon,
- permitted action modes.

---

# 8. Goal Hierarchy

The planner should reason hierarchically.

Example:

```text
GLOBAL
Obtain a diamond

    LONG-TERM
    Obtain an iron pickaxe

        SHORT-TERM
        Obtain iron ingots

            LOCAL
            Search for exposed iron

            LOCAL
            Mine iron

            LOCAL
            Smelt iron

        SHORT-TERM
        Craft iron pickaxe

    LONG-TERM
    Reach suitable mining depth

    LONG-TERM
    Find diamond ore

    LONG-TERM
    Mine diamond
```

Every executable node should have a machine-checkable verifier.

The LLM decides which node to activate.

The RL executor solves the active node.

This drastically reduces long-horizon credit assignment.

---

# 9. Skill Ontology

Start with a small set of semantic goal categories.

Example:

```text
EXPLORE
SEARCH
APPROACH
NAVIGATE
LOOK_AT
MINE
COLLECT
PLACE
INTERACT
ATTACK
ESCAPE
EQUIP
SELECT_HOTBAR
CRAFT
SMELT
EAT
WAIT
```

These are not separate hard-coded policies.

They are structured task categories used to condition a general policy.

Example:

```text
skill = MINE
target = minecraft:stone
```

and:

```text
skill = MINE
target = minecraft:iron_ore
```

should use the same general executor.

---

# 10. Verifier DSL

The verifier DSL is one of the most important pieces of the entire project.

It should represent deterministic predicates.

Initial predicates:

```text
InventoryAtLeast(item, count)
InventoryDelta(item, count)

Crafted(item)
Smelted(item)

BlockBroken(block)
BlockPlaced(block)

EntityKilled(entity)
EntityDamaged(entity)

ItemUsed(item)
ItemEquipped(item)

Near(target, radius)
PositionWithin(region)

LookingAt(target)
TargetVisible(target)

HealthAbove(value)
HealthBelow(value)
FoodAbove(value)

NoHealthLoss()
FallDistanceBelow(value)

AtDepth(y)
DepthBelow(y)
DepthAbove(y)

EventOccurred(event)

TimeBelow(seconds)
StepCountBelow(steps)
```

Logical composition:

```text
AND(...)
OR(...)
NOT(...)
SEQUENCE(...)
ANY(...)
ALL(...)
```

Example:

```text
AND(
    InventoryAtLeast("minecraft:iron_ingot", 3),
    HealthAbove(4)
)
```

Each task specification should contain:

```text
preconditions
success
failure
progress
timeout
```

The LLM may generate or parameterize task specifications.

The verifier evaluates them deterministically.

---

# 11. Reward Design

The primary optimization target should be exact task success.

Use sparse success rewards plus conservative shaping.

General form:

\[
r_t =
R_{\text{success}} \mathbf{1}[\text{success}]
+
\lambda \left(
\gamma \Phi(s_{t+1}) - \Phi(s_t)
\right)
-
c_{\text{step}}
+
r_{\text{safety}}
\]

where:

- \(R_{\text{success}}\) is the dominant terminal reward,
- \(\Phi\) is a task-specific progress potential,
- \(c_{\text{step}}\) discourages pointless behavior,
- \(r_{\text{safety}}\) penalizes unsafe events where relevant.

Possible progress terms:

### Navigation

```text
negative distance to target
target visibility
angular alignment
```

### Mining

```text
target detected
target centered
target in reach
correct tool selected
block damage progress
block broken
item collected
```

### Crafting

```text
required materials available
crafting interface open
recipe selected
output obtained
```

Reward shaping should remain weak enough that the exact success predicate remains the objective.

---

# 12. Executor Architecture

## 12.1 High-level design

Use a recurrent, goal-conditioned, multimodal policy.

Inputs:

```text
RGB / video
structured player state
inventory
current goal
visual target representation
recent actions
recent events
temporal memory
```

Outputs:

```text
raw Minecraft action
or
high-level action option

STOP estimate
FAILURE estimate
value estimate
task progress
target visibility
optional target location
```

---

# 13. Visual Encoder

Start with a Minecraft-pretrained visual and motor representation.

Recommended initialization:

```text
VPT-style MineStudio checkpoint
preferably VPT 2x or closest strong available checkpoint
```

Reason:

A Minecraft-pretrained policy already understands:

- movement,
- mouse control,
- common visual affordances,
- block interaction,
- inventory interaction,
- camera behavior,
- Minecraft-specific appearance.

This is significantly more useful than starting from a generic vision model alone.

## Optional DINO branch

A DINO-family visual encoder can be added later as a second branch for stronger dense semantics.

Possible architecture:

```text
RGB
 | \
 |  \
 |   +--> DINO spatial features --------+
 |                                      |
 +-----> VPT visual features -----------+
                                        |
structured state ----------------------+
                                        |
goal tokens ---------------------------+
                                        |
visual goal tokens --------------------+
                                        |
                                multimodal fusion
                                        |
                              recurrent transformer
```

Ablate:

```text
VPT only
vs
VPT + DINO
```

Do not make DINO mandatory in V1.

---

# 14. Structured State Encoder

Useful actor-visible state may include:

```text
health
hunger
armor
air
experience
position
velocity if available
yaw
pitch
current hotbar slot
inventory
equipped item
GUI state
recent game events
time of day
biome if experimentally allowed
```

Inventory should be tokenized rather than flattened.

For each slot:

```text
item embedding
count embedding
slot embedding
durability embedding
metadata embedding
```

Then use attention across inventory tokens.

---

# 15. Goal Encoder

The goal representation should combine:

```text
skill ID embedding
text embedding
structured parameters
target identity
success/failure specification embedding
visual grounding information
```

Language provides open-vocabulary flexibility.

Structured fields provide stable semantics.

The policy should not be forced to infer every action category from arbitrary wording.

---

# 16. Visual Grounding

Depending on task type, the goal may contain:

```text
image point
segmentation mask
bounding box
reference crop
entity ID
text-only semantic target
```

Examples:

### Mine this block

Use:

```text
block identity
image point or mask
```

### Go to that mountain

Use:

```text
image region or target point
```

### Find a cow

Initially:

```text
semantic target = cow
```

Once detected:

```text
bind target to detected object
```

Exact spatial grounding should be treated as first-class information.

---

# 17. Temporal Memory

Minecraft is partially observable.

The executor therefore requires temporal memory.

Recommended first implementation:

```text
Transformer-XL style recurrent memory
or
causal transformer with cached history
```

Initial history horizon:

```text
128 to 256 policy steps
```

This can later be extended with:

- compressed episodic memory,
- object memory,
- spatial memory,
- learned map tokens.

---

# 18. Policy Outputs

The executor should support two action granularities.

## Raw control

Examples:

```text
forward
back
left
right
jump
sprint
sneak

attack
use

hotbar selection

camera dx
camera dy
```

## Closed-loop action options

Examples:

```text
ATTACK_UNTIL(
    block_break |
    entity_dead |
    target_lost |
    timeout
)

USE_UNTIL(
    consumption_complete |
    interaction_complete |
    timeout
)

SELECT_HOTBAR(slot)

CENTER_CAMERA_ON(target)

MOVE_TOWARD(target)

INTERACT(target)
```

The policy should decide whether to use:

```text
raw actions
or
an action option
```

This allows adaptive action granularity.

---

# 19. Why MPC Is Not Required in V1

Do not include MPC in the first implementation.

Minecraft primarily requires:

- discrete key control,
- visual camera control,
- short reaction loops,
- event-driven interaction.

It does not provide the kind of continuous actuator dynamics where classical MPC is most useful.

A recurrent visuomotor policy already acts as a learned closed-loop controller.

For simple visual alignment, direct visual servoing is enough.

Example:

\[
\Delta \theta_t = k_x(x_{\text{target}} - x_{\text{center}})
\]

\[
\Delta \phi_t = k_y(y_{\text{target}} - y_{\text{center}})
\]

MPC can later be investigated as a research ablation if a learned world model becomes available.

It is not part of the critical path.

---

# 20. LLM Planner

The multimodal LLM should operate at a low frequency or event-driven cadence.

It should not run at policy frequency.

Recommended triggers:

```text
new global objective
goal success
goal failure
goal timeout
unexpected inventory change
unexpected damage
danger detection
executor requests replan
major environment event
periodic low-frequency checkpoint
```

Typical planner frequency may be on the order of seconds, not frames.

---

# 21. Planner Tools

The LLM should interact through tools.

Initial tool set:

```text
observe_current_frame()

observe_recent_video(seconds)

get_inventory()

get_player_state()

get_recent_events()

get_active_goal()

get_goal_status()

get_goal_history()

recipe(item)

retrieve_world_memory(query)

ground_target(description)

get_target_crop()

get_target_mask()

get_known_locations()

get_local_map()
```

Some tools may expose privileged information only during training.

The planner API must respect information privilege rules.

---

# 22. World Memory

The planner should maintain persistent environment knowledge.

Examples:

```text
base location
known cave entrances
known resource locations
furnaces
crafting tables
dangerous zones
recent deaths
unreachable targets
previous failed plans
known landmarks
```

Memory should distinguish:

```text
observed fact
inferred fact
stale fact
uncertain fact
```

Possible world-memory entries:

```json
{
  "type": "location",
  "label": "cave_entrance_02",
  "position": [125.2, 68.0, -31.4],
  "confidence": 0.96,
  "source": "observed",
  "last_seen_step": 4812
}
```

---

# 23. Behavior Cloning Data Collection

MineStudio should be used to record human play directly.

Record at approximately:

```text
10 to 20 Hz
```

Each transition should eventually contain:

```text
episode_id
frame_id
timestamp

rgb
action

position
yaw
pitch

health
food
armor

inventory
equipped_item

events

global_goal_id
plan_id
subgoal_id
skill_id
goal_packet_id

human_controlled
policy_controlled
intervention

success
failure
progress
```

Do not require all metadata to be manually entered during gameplay.

---

# 24. Human Data Collection Modes

Use three modes.

## Mode A: pure demonstration

Human controls the player for the entire episode.

Useful for:

- bootstrapping,
- new skills,
- long coherent demonstrations.

## Mode B: agent with human takeover

The policy controls the player.

The human presses a takeover key when the policy makes a mistake.

Record:

```text
several seconds before takeover
human correction
several seconds after recovery
```

This is DAgger-style intervention data.

It is disproportionately valuable because it targets the current policy's failure distribution.

## Mode C: human-assisted planning

The policy executes goals selected by the planner.

A human may:

- modify the planner goal,
- reject a bad target,
- provide a better grounding,
- manually choose a recovery objective.

These corrections can later supervise the planner.

---

# 25. Human Annotation

Avoid continuous manual annotation.

Use three annotation sources.

## Explicit annotation

Optional hotkeys:

```text
START_GOAL
END_GOAL
NEXT_GOAL
FAILURE
TAKEOVER
```

Optional compact UI for selecting:

```text
approach
mine
craft
collect
fight
escape
```

## Environment-derived annotation

Use events such as:

```text
block broken
item collected
item crafted
item smelted
entity killed
damage received
inventory changed
```

These provide exact task boundaries.

## Offline automatic annotation

For each trajectory segment provide the LLM:

```text
initial frame
final frame
short video clip
initial inventory
final inventory
event list
action summary
```

Ask the LLM to propose:

```text
goal category
goal description
target identity
success predicate
```

Then validate the proposed goal using the deterministic verifier.

---

# 26. Hindsight Goal Relabeling

Every human trajectory should generate multiple training examples.

Example human sequence:

```text
walk toward tree
look at tree
equip axe
break log
collect log
craft planks
```

Possible automatically extracted goals:

```text
approach tree
look at tree
select axe
mine oak log
collect oak log
obtain oak log
craft planks
obtain planks
```

This dramatically multiplies the value of recorded demonstrations.

The environment event stream should be used to identify candidate hindsight goals.

---

# 27. Training Pipeline

The recommended training progression is:

```text
Minecraft pretrained policy
        |
        v
goal-conditioned behavior cloning
        |
        v
hindsight-relabeled behavior cloning
        |
        v
agent rollout + human interventions
        |
        v
DAgger-style retraining
        |
        v
procedural task curriculum
        |
        v
KL-regularized PPO
        |
        v
long-horizon hierarchical evaluation
```

---

# 28. Phase 0: Pretrained Minecraft Policy

Do not start from random weights.

Initialize from a strong Minecraft behavior model.

Recommended:

```text
MineStudio VPT-compatible checkpoint
```

Initially freeze much of the visual and motor backbone.

Add new modules for:

```text
goal conditioning
structured state
visual grounding
value head
progress head
failure head
```

Then gradually unfreeze.

---

# 29. Phase 1: Goal-Conditioned Behavior Cloning

Train:

\[
\mathcal{L}_{BC}
=
-\sum_t
\log \pi_\theta(a_t | o_{\leq t}, s_{\leq t}, g_t)
\]

where:

- \(o_t\): visual observation,
- \(s_t\): structured environment state,
- \(g_t\): GoalPacket,
- \(a_t\): human action.

Use sequence training, not isolated frames.

---

# 30. Auxiliary Losses

Recommended:

\[
\mathcal{L}
=
\mathcal{L}_{action}
+
\lambda_v \mathcal{L}_{visibility}
+
\lambda_p \mathcal{L}_{target}
+
\lambda_e \mathcal{L}_{event}
+
\lambda_g \mathcal{L}_{progress}
+
\lambda_s \mathcal{L}_{stop}
+
\lambda_f \mathcal{L}_{failure}
\]

Possible targets:

```text
target visible
target location
target mask
next environment event
goal progress
goal completed
goal impossible
need replan
```

These auxiliary tasks should improve spatial grounding and representation learning.

---

# 31. Phase 2: DAgger / Human Intervention Training

Behavior cloning suffers from covariate shift.

A small mistake sends the policy into states not represented in human demonstrations.

Use interactive collection:

```text
agent acts
   |
   +--> correct behavior
   |
   +--> mistake
          |
          v
      human takeover
          |
          v
       recovery
```

Add intervention transitions to the dataset.

Retrain.

Repeat.

Track:

```text
interventions per minute
intervention duration
recovery success
intervention type
```

The intervention rate should become a primary training metric.

---

# 32. Phase 3: Procedural Curriculum

Do not manually create a fixed task list.

Create task generators.

Example:

```text
Mine(block_type, amount)
```

Parameter ranges:

```text
block_type:
    dirt
    wood
    stone
    coal
    iron
    ...

amount:
    1
    2
    4
    8
```

Randomize:

```text
world seed
biome
lighting
distance
orientation
terrain
target visibility
inventory
available tools
health
hunger
distractors
nearby enemies
starting position
```

Curriculum scheduler tracks success by task family.

Possible policy:

```text
success > 90%:
    increase difficulty

success between 20% and 90%:
    continue sampling heavily

success < 20%:
    reduce difficulty or request human demos
```

Do not permanently remove solved tasks.

Continue mixing old tasks to avoid forgetting.

---

# 33. Phase 4: Online RL

Use online RL only after obtaining a competent imitation policy.

Recommended initial algorithm:

```text
PPO
```

with KL regularization to the behavior-cloned reference policy.

Conceptual objective:

\[
\mathcal{L}
=
\mathcal{L}_{PPO}
+
\beta D_{KL}(\pi_\theta || \pi_{BC})
+
\lambda_{BC}\mathcal{L}_{replay}
\]

The KL term prevents catastrophic drift.

The replay term continuously anchors the policy to strong human behavior.

RL should refine behavior, not rediscover basic Minecraft control.

---

# 34. Why Not Start With Offline RL

Offline RL may be tested later.

Possible algorithms:

```text
IQL
AWAC
AWR
advantage-weighted BC
```

However, the project has access to a live simulator.

The simpler path is:

```text
pretraining
BC
DAgger
online PPO
```

This should be implemented before adding complex off-policy value estimation.

Offline RL remains a useful ablation.

---

# 35. State Coverage and Statistical Voids

RL does not automatically solve missing demonstration coverage.

Coverage should be addressed with:

1. Minecraft-pretrained behavior priors
2. procedural environment variation
3. hindsight relabeling
4. DAgger-style interventions
5. on-policy RL
6. recovery demonstrations
7. failure prediction
8. uncertainty estimation
9. KL regularization
10. continued demonstration replay

Particularly valuable demonstrations include:

```text
recover from hole
recover from water
recover after losing target
recover after taking damage
recover after tool breaks
recover after inventory full
recover after unexpected enemy
recover from bad camera orientation
recover after failed jump
```

Failure recovery should be deliberately collected.

---

# 36. Planner / Executor Interaction

The planner should not micromanage every frame.

Example:

```text
LLM:
    "Mine the visible stone block on the right."

GoalPacket:
    skill = MINE
    target = stone
    target_point = ...
    success = inventory_delta(cobblestone, +1)

Executor:
    orient
    approach
    select correct tool
    mine
    collect

Verifier:
    cobblestone +1

Result:
    SUCCESS
```

The executor can operate for multiple seconds without consulting the LLM.

---

# 37. Replanning Conditions

The executor should be able to request replanning.

Possible reasons:

```text
target lost
target unreachable
unexpected danger
goal impossible
missing required tool
inventory full
low health
large navigation failure
stuck state
timeout
uncertainty too high
environment changed
```

Example response:

```json
{
  "status": "REPLAN",
  "reason": "required_tool_missing",
  "details": {
    "required_tool": "minecraft:iron_pickaxe"
  }
}
```

---

# 38. Adaptive Action Granularity

The executor should choose between:

```text
raw controls
parameterized action options
```

Example where an option is appropriate:

```text
Eat cooked beef
```

Possible execution:

```text
select slot
hold use
stop when food consumed
```

Example where raw visuomotor policy is required:

```text
Escape from zombies and climb to higher terrain
```

The architecture should allow both.

---

# 39. Action Options as Semi-Markov Controls

An action option has:

```text
initiation condition
policy/controller
termination condition
timeout
```

Example:

```text
ATTACK_UNTIL_BLOCK_BREAK
```

Initiation:

```text
target block visible
target within reach
```

Execution:

```text
hold attack
maintain camera alignment
```

Termination:

```text
block break event
target lost
timeout
```

This reduces unnecessary policy burden while preserving closed-loop control.

---

# 40. Planner Safety and Hallucination Prevention

The LLM should not directly execute arbitrary Minecraft commands.

Instead:

```text
LLM
 |
 v
GoalPacket / SkillSpec
 |
 v
schema validation
 |
 v
precondition validation
 |
 v
verifier compilation
 |
 v
executor
```

If compilation fails:

```text
return structured error to LLM
```

Example:

```json
{
  "error": "INVALID_GOAL",
  "reason": "success predicate references unknown item",
  "field": "success.item"
}
```

This prevents linguistically plausible but undefined behavior.

---

# 41. TaskSpec

Use a formal task object.

Example:

```json
{
  "task_id": "mine_iron_001",
  "family": "mine_block",
  "initialization": {
    "world_generator": "random_survival",
    "inventory": {
      "minecraft:stone_pickaxe": 1
    }
  },
  "goal": {
    "skill": "MINE",
    "target": "minecraft:iron_ore"
  },
  "success": {
    "predicate": "InventoryDelta",
    "args": ["minecraft:raw_iron", 1]
  },
  "failure": {
    "predicate": "OR",
    "children": [
      {
        "predicate": "HealthBelow",
        "args": [2]
      },
      {
        "predicate": "StepCountAbove",
        "args": [1200]
      }
    ]
  }
}
```

TaskSpec should be independent of MineStudio.

Environment adapters compile TaskSpec into environment-specific setup.

---

# 42. Dataset Schema

Recommended logical schema:

```text
Episode
    episode_id
    environment
    version
    seed
    task_spec
    global_goal
    outcome

Transition
    episode_id
    timestep
    timestamp

    observation
    rgb_path

    structured_state
    inventory
    events

    action_raw
    action_option

    goal_packet

    control_source
        human
        policy
        recovery
        scripted

    intervention
    reward
    progress
    success
    failure
```

Large binary observations should not be stored directly inside Parquet.

Store references to video/frame storage.

---

# 43. Recommended Repository Structure

```text
minecraft_agent/
|
|-- PROJECT.md
|-- README.md
|-- pyproject.toml
|
|-- configs/
|   |-- environment/
|   |-- model/
|   |-- tasks/
|   |-- training/
|   `-- evaluation/
|
|-- minecraft_agent/
|   |
|   |-- env/
|   |   |-- base.py
|   |   |-- minestudio_env.py
|   |   |-- craftground_env.py
|   |   `-- observation.py
|   |
|   |-- goals/
|   |   |-- schemas.py
|   |   |-- compiler.py
|   |   `-- ontology.py
|   |
|   |-- verifier/
|   |   |-- predicates.py
|   |   |-- evaluator.py
|   |   |-- progress.py
|   |   `-- registry.py
|   |
|   |-- planner/
|   |   |-- agent.py
|   |   |-- tools.py
|   |   |-- prompts.py
|   |   |-- memory.py
|   |   `-- schemas.py
|   |
|   |-- perception/
|   |   |-- grounding.py
|   |   |-- segmentation.py
|   |   `-- tracking.py
|   |
|   |-- policy/
|   |   |-- model.py
|   |   |-- vision.py
|   |   |-- state_encoder.py
|   |   |-- goal_encoder.py
|   |   |-- memory.py
|   |   |-- heads.py
|   |   `-- action_space.py
|   |
|   |-- options/
|   |   |-- base.py
|   |   |-- mining.py
|   |   |-- use.py
|   |   |-- camera.py
|   |   `-- navigation.py
|   |
|   |-- data/
|   |   |-- recorder.py
|   |   |-- dataset.py
|   |   |-- relabel.py
|   |   |-- segmentation.py
|   |   `-- metadata.py
|   |
|   |-- training/
|   |   |-- bc.py
|   |   |-- dagger.py
|   |   |-- ppo.py
|   |   |-- losses.py
|   |   `-- replay.py
|   |
|   |-- curriculum/
|   |   |-- generator.py
|   |   |-- scheduler.py
|   |   `-- difficulty.py
|   |
|   `-- evaluation/
|       |-- runner.py
|       |-- metrics.py
|       `-- benchmark.py
|
|-- scripts/
|   |-- play_and_record.py
|   |-- collect_interventions.py
|   |-- train_bc.py
|   |-- train_rl.py
|   |-- run_agent.py
|   |-- evaluate.py
|   `-- inspect_episode.py
|
|-- tests/
|   |-- test_verifier.py
|   |-- test_goal_compiler.py
|   |-- test_task_specs.py
|   `-- test_environment.py
|
`-- notebooks/
    |-- dataset_analysis.ipynb
    |-- policy_debug.ipynb
    `-- evaluation_analysis.ipynb
```

---

# 44. Required Interfaces

## Environment

```python
class MinecraftEnvironment:
    def reset(self, task_spec):
        ...

    def step(self, action):
        ...

    def get_observation(self):
        ...

    def get_verifier_state(self):
        ...

    def execute_setup(self, task_spec):
        ...
```

## Verifier

```python
class Verifier:
    def evaluate(self, task_spec, previous_state, current_state):
        ...
```

Expected output:

```text
success
failure
progress
reward_components
events
```

## Planner

```python
class Planner:
    def plan(self, context):
        ...
```

Output:

```text
GoalPacket
```

## Executor

```python
class Executor:
    def act(self, observation, goal_packet, memory):
        ...
```

Output:

```text
raw action
or action option
```

---

# 45. Core Metrics

Never evaluate only cumulative reward.

Track:

```text
task success rate
subgoal success rate
global objective success rate

steps to completion
seconds to completion

replan count
replans per minute

human intervention count
human intervention duration

recovery success rate

death rate
health loss
fall events

invalid planner goal rate
verifier compilation failure rate

target grounding accuracy
target visibility accuracy

held-out seed success
held-out biome success
held-out task composition success

action efficiency
inventory efficiency
```

---

# 46. Evaluation Splits

Evaluation must prevent memorization.

Separate:

```text
training seeds
validation seeds
test seeds
```

Also create:

```text
held-out biome splits
held-out terrain distributions
held-out inventory conditions
held-out target distances
held-out lighting conditions
held-out task compositions
```

Long-horizon generalization should be evaluated separately from local skill competence.

---

# 47. Mandatory Ablations

At minimum:

| Experiment | Question |
|---|---|
| VPT vs VPT + DINO | Does generic dense vision improve grounding? |
| text goal vs structured goal | Does typed intent improve control? |
| text vs text + visual target | Does grounding improve interaction? |
| raw actions vs raw + options | Does adaptive granularity improve performance? |
| BC vs BC + hindsight | How much supervision can be extracted automatically? |
| BC vs BC + DAgger | How important is covariate shift? |
| BC vs BC + PPO | Does RL improve beyond demonstrations? |
| PPO vs KL-PPO | Does KL anchoring prevent policy collapse? |
| RGB only vs RGB + structured state | How useful is privileged-but-allowed state? |
| symmetric vs asymmetric critic | Does privileged critic information improve learning? |
| periodic vs event-driven LLM | How often does the planner need to run? |
| fixed skills vs adaptive action granularity | Does action abstraction need to be contextual? |

---

# 48. Primary Research Questions

The project should aim to answer:

1. Can long-horizon Minecraft behavior be made substantially easier by converting it into a sequence of automatically verifiable local objectives?

2. How much does behavior cloning reduce RL sample complexity?

3. How important are intervention demonstrations compared with passive demonstrations?

4. Does exact visual grounding significantly improve low-level task execution?

5. Is structured goal conditioning substantially better than natural-language-only conditioning?

6. Can adaptive action granularity outperform a fixed raw action space?

7. How much online RL improvement is possible without destroying pretrained visuomotor competence?

8. Can a multimodal LLM planner reliably decompose unseen Minecraft objectives into executor-solvable tasks?

9. Can hindsight relabeling turn ordinary human gameplay into a large multi-task supervision dataset?

10. How well does the system generalize to unseen world seeds, terrain, and task compositions?

---

# 49. Initial Task Curriculum

Start simple.

## Tier 0: motor sanity

```text
look left/right
look at target
walk forward
walk to visible point
jump onto one block
select hotbar slot
attack visible block
use selected item
```

## Tier 1: local interaction

```text
mine visible dirt
mine visible stone
mine visible log
collect dropped item
place block at target
eat food
attack nearby passive mob
```

## Tier 2: short compositional tasks

```text
approach tree + mine log
find visible stone + mine stone
select correct tool + mine block
collect item + craft simple recipe
navigate around obstacle
recover after losing target
```

## Tier 3: resource tasks

```text
obtain wood
obtain cobblestone
obtain coal
obtain raw iron
smelt iron
craft stone pickaxe
craft iron pickaxe
```

## Tier 4: longer objectives

```text
start from empty inventory -> stone tools
obtain food
build furnace
obtain iron pickaxe
create shelter
```

## Tier 5: global Minecraft objectives

```text
obtain diamond
enter Nether
obtain blaze rods
reach End
defeat dragon
```

The project should not begin with Tier 5.

---

# 50. V1 Scope

V1 should prove the executor and data pipeline before solving full Minecraft.

V1 success criteria:

```text
MineStudio environment running reliably
human play recording works
GoalPacket schema implemented
verifier DSL implemented
5 to 10 task families implemented
VPT-based goal-conditioned executor implemented
behavior cloning works
human takeover recording works
hindsight relabeling works
held-out task evaluation works
KL-PPO improves at least some tasks beyond BC
```

Example V1 benchmark:

```text
Mine visible block
Approach target
Collect target
Equip requested tool
Mine requested block
Eat food
Escape danger
Navigate to visible target
```

---

# 51. V2 Scope

Add:

```text
multimodal LLM planner
world memory
visual grounding
longer compositional goals
automatic planner evaluation
procedural curricula
adaptive action options
asymmetric critic
large-scale parallel RL
```

---

# 52. V3 Scope

Potential extensions:

```text
learned world model
model-based short-horizon control
MPC ablation
3D spatial memory
semantic mapping
multi-agent Minecraft
language-conditioned world modeling
self-generated curricula
automatic skill discovery
skill compression
cross-environment transfer
Bedrock-RL massive-scale training
```

---

# 53. Failure Modes to Expect

## Reward hacking

Mitigation:

```text
exact verifier
multiple predicates
anti-cheat validation
held-out evaluation
```

## LLM hallucinated goals

Mitigation:

```text
strict schema
goal compiler
precondition checking
task ontology
structured error feedback
```

## Policy covariate shift

Mitigation:

```text
DAgger
human interventions
RL
recovery demonstrations
```

## Catastrophic forgetting during RL

Mitigation:

```text
KL to BC reference
human replay
mixed curriculum
frozen backbone phases
```

## Overfitting to world seeds

Mitigation:

```text
procedural seeds
seed-disjoint test set
biome split
terrain split
```

## Planner over-control

Mitigation:

```text
event-driven planning
minimum executor horizon
avoid frame-level replanning
```

## Executor under-control

Mitigation:

```text
failure prediction
replan requests
timeouts
progress monitoring
```

## Privileged information leakage

Mitigation:

```text
strict observation API
separate verifier state
separate actor state
tests for forbidden fields
```

---

# 54. Engineering Priorities

The order matters.

Implement:

```text
1. environment wrapper
2. task/verifier DSL
3. human recorder
4. dataset format
5. pretrained policy loading
6. goal-conditioned executor
7. BC
8. evaluation harness
9. hindsight relabeling
10. human intervention collection
11. DAgger retraining
12. curriculum generation
13. PPO
14. LLM planner
15. visual grounding
16. long-horizon planning
```

Do not start by building the LLM planner.

Without the task/verifier/executor substrate, planner quality is impossible to measure.

---

# 55. Minimal First Experiment

A useful first complete experiment:

## Task

```text
Mine one visible stone block.
```

## Environment

Randomized:

```text
camera orientation
stone location
distance
lighting
terrain
hotbar configuration
```

## Input

```text
RGB
inventory
selected hotbar slot
goal = MINE stone
optional image point
```

## Success

```text
InventoryDelta("minecraft:cobblestone", 1)
```

## Training

```text
VPT initialization
human demos
BC
hindsight augmentation
intervention data
PPO
```

## Compare

```text
BC only
BC + intervention
BC + PPO
BC + intervention + KL-PPO
```

If this task cannot be made highly reliable, do not scale to long-horizon objectives yet.

---

# 56. Second Experiment

## Task

```text
Obtain cobblestone.
```

The target is not initially grounded.

The planner/executor must:

```text
explore
detect visible stone
bind target
approach
select tool
mine
collect
```

This tests the transition from semantic goals to grounded local control.

---

# 57. Third Experiment

## Task

```text
Obtain a stone pickaxe.
```

The planner must decompose:

```text
obtain wood
craft planks
craft sticks
obtain stone
craft stone pickaxe
```

Each subgoal is independently verifiable.

This becomes the first serious hierarchical evaluation.

---

# 58. Key Design Principles

These principles should guide all future implementation.

## Principle 1

**LLM reasoning and motor control must remain separate.**

## Principle 2

**Every executable goal should have an exact verifier whenever Minecraft state permits it.**

## Principle 3

**Use structured goals, not language alone.**

## Principle 4

**Exploit human data before relying on RL exploration.**

## Principle 5

**Collect intervention and recovery data, not just successful demonstrations.**

## Principle 6

**Use pretrained Minecraft visuomotor knowledge.**

## Principle 7

**Use online RL to improve a competent policy, not to bootstrap one from nothing.**

## Principle 8

**Allow different action granularities.**

## Principle 9

**Do not expose privileged information accidentally.**

## Principle 10

**Evaluate on held-out world distributions, not only training tasks.**

---

# 59. Final Recommended V1 Stack

```text
Environment:
    MineStudio

Platform:
    native Ubuntu Linux

Policy initialization:
    MineStudio VPT-style checkpoint
    preferably VPT 2x

Executor:
    recurrent goal-conditioned transformer

Visual input:
    RGB
    VPT visual features
    optional later DINO branch

Structured input:
    health
    hunger
    inventory
    equipped item
    position
    yaw/pitch
    events

Goal input:
    structured GoalPacket
    text
    skill ID
    target identity
    optional point/mask/crop

Action space:
    raw VPT-style controls
    +
    closed-loop action options

Planner:
    multimodal LLM
    event-driven
    tool-based
    structured output

Verifier:
    deterministic TaskSpec / predicate DSL

Human data:
    MineStudio direct play recording
    human takeover
    hindsight relabeling

Training:
    pretrained policy
    ->
    behavior cloning
    ->
    hindsight BC
    ->
    DAgger/interventions
    ->
    procedural curriculum
    ->
    KL-regularized PPO

Tracking:
    Weights & Biases

Distributed execution:
    Ray

Metadata:
    Parquet + DuckDB/Polars

MPC:
    not used in V1
```

---

# 60. Final Project Statement

The project is not:

> "Train an RL agent to play Minecraft."

It is:

> **Build a hierarchical Minecraft intelligence system in which a multimodal planner converts long-horizon semantic objectives into grounded, automatically verifiable control problems, and a pretrained visuomotor policy learns to solve those problems through human imitation, intervention learning, and reinforcement learning.**

The main scientific bet is that Minecraft becomes tractable when the semantic planning problem, the local control problem, and the evaluation problem are explicitly separated.

The LLM handles:

```text
why
what
what next
```

The learned executor handles:

```text
how
```

The verifier handles:

```text
did it actually happen
```

Human demonstrations provide:

```text
a strong initial behavioral manifold
```

RL provides:

```text
adaptation beyond the demonstrations
```

The most important implementation order is:

```text
verifier + data collection
        ->
goal-conditioned executor
        ->
behavior cloning
        ->
intervention training
        ->
online RL
        ->
LLM planning
        ->
long-horizon Minecraft
```

That order keeps the research measurable at every stage and prevents the project from becoming an untestable collection of interacting agent components.
