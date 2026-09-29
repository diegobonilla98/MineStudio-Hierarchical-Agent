import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
FIGURES = ROOT / "figures"
OUTPUT = ROOT.parent / "output" / "pdf" / "system_0_1_2_minecraft_agent_study.pdf"
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_LEFT = 1.75 * cm
MARGIN_RIGHT = 1.75 * cm
MARGIN_TOP = 1.65 * cm
MARGIN_BOTTOM = 1.55 * cm
BLUE = colors.HexColor("#165D8C")
CYAN = colors.HexColor("#3EA6B8")
ORANGE = colors.HexColor("#D97732")
INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#5E6A75")
LIGHT = colors.HexColor("#EEF4F7")
RED = colors.HexColor("#A63D40")


def load_json(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def complete_rows(name):
    rows = []
    for line in (EVIDENCE / name).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("status") == "complete":
            rows.append(row)
    return rows


def milestone_value(row, milestone):
    entry = row.get("milestones", {}).get(milestone, {})
    return entry.get("step") is not None


def common_milestone_rates():
    full = {row["seed"]: row for row in complete_rows("iron_full_results.jsonl")}
    no_specialist = {row["seed"]: row for row in complete_rows("iron_no_specialist_results.jsonl")}
    no_system2 = {row["seed"]: row for row in complete_rows("iron_no_system2_results.jsonl")}
    seeds = sorted(set(full) & set(no_specialist) & set(no_system2))
    milestones = [
        "logs_obtained",
        "wooden_pickaxe",
        "required_cobblestone",
        "stone_pickaxe",
        "furnace_available",
        "raw_iron_3",
        "iron_ingots_3",
        "iron_pickaxe",
    ]
    rates = {}
    for arm, rows in [("Full", full), ("No specialist", no_specialist), ("No System 2", no_system2)]:
        rates[arm] = [sum(milestone_value(rows[seed], milestone) for seed in seeds) / len(seeds) for milestone in milestones]
    return seeds, milestones, rates


def save_architecture():
    path = FIGURES / "architecture.png"
    fig, axis = plt.subplots(figsize=(10.5, 5.1))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 6)
    axis.axis("off")
    boxes = [
        (3.25, 4.75, 3.5, 0.8, "SYSTEM 2", "Gemini: plan, interpret, replan", "#164E70"),
        (3.25, 3.45, 3.5, 0.72, "EXECUTION ROUTER", "structured objective and handoff", "#357A8B"),
        (0.9, 1.85, 3.2, 1.05, "SYSTEM 0", "verified deterministic mechanics", "#D97732"),
        (5.9, 1.85, 3.2, 1.05, "SYSTEM 1", "STEVE-1 plus task specialist", "#2B7A78"),
        (3.25, 0.35, 3.5, 0.8, "MINESTUDIO", "pixels, actions, exact verifier", "#4D5966"),
    ]
    for x, y, width, height, title, subtitle, color in boxes:
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.09", facecolor=color, edgecolor="none")
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height * 0.62, title, ha="center", va="center", color="white", fontsize=14, fontweight="bold")
        axis.text(x + width / 2, y + height * 0.27, subtitle, ha="center", va="center", color="white", fontsize=9.5)
    arrows = [((5, 4.75), (5, 4.17)), ((5, 3.45), (2.5, 2.9)), ((5, 3.45), (7.5, 2.9)), ((2.5, 1.85), (4.35, 1.15)), ((7.5, 1.85), (5.65, 1.15)), ((5, 0.35), (5, 0.02))]
    for start, end in arrows:
        axis.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color="#3B4652", lw=1.8))
    axis.text(5, 5.85, "Proposed decomposition", ha="center", va="center", fontsize=17, fontweight="bold", color="#17212B")
    fig.tight_layout()
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def save_behavioral_chart(behavioral):
    path = FIGURES / "behavioral_screen.png"
    selected = ["vanilla", "upper_lora_r8_step4800", "upper_lora_r8_step6400", "upper_lora_r32_step6400", "recurrent_upper_lora_step6400"]
    labels = ["Vanilla", "r8 / 4800", "r8 / 6400", "r32 / 6400", "Recurrent + upper"]
    records = {row["policy"]: row for row in behavioral["policies"]}
    normal = [records[name]["normal_success_rate"] * 100 for name in selected]
    hazard = [records[name]["hazard_success_rate"] * 100 for name in selected]
    combined = [records[name]["combined_success_rate"] * 100 for name in selected]
    positions = range(len(labels))
    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    width = 0.24
    axis.bar([x - width for x in positions], normal, width, label="Normal", color="#71A8C4")
    axis.bar(positions, hazard, width, label="Hazard", color="#D98D58")
    axis.bar([x + width for x in positions], combined, width, label="Combined", color="#2B7A78")
    axis.set_xticks(list(positions), labels)
    axis.set_ylim(0, 90)
    axis.set_ylabel("Success rate (%)")
    axis.set_title("Behavioral selection disagreed with offline loss")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    for index, value in enumerate(combined):
        axis.text(index + width, value + 1.5, f"{value:.0f}%", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def save_boundary_chart(isolation):
    path = FIGURES / "system0_boundary.png"
    metrics = isolation["combined"]["metrics"]
    names = ["old_router_old_pickup", "old_router_hardened_pickup", "new_router_old_pickup", "new_router_hardened_pickup"]
    labels = ["Old / old", "Old / hardened", "New / old", "New / hardened"]
    success = [metrics[name]["success_rate"] * 100 for name in names]
    pickup = [metrics[name]["strict_pickup_tail_failures"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.4, 1]})
    axes[0].bar(labels, success, color=["#2B7A78", "#80A8B1", "#71A8C4", "#D98D58"])
    axes[0].set_ylim(0, 75)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("End-to-end stone acquisition")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].tick_params(axis="x", rotation=18)
    axes[1].bar(labels, pickup, color=["#A63D40", "#D98D58", "#71A8C4", "#2B7A78"])
    axes[1].set_ylabel("Strict pickup-tail failures")
    axes[1].set_title("Local defect removal")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Fixing a local deterministic defect degraded the interface", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def save_micro_chart(micro, ppo):
    path = FIGURES / "micro_and_ppo.png"
    order = ["EXIT_WATER", "REACQUIRE_STONE", "CLIMB_SHORE", "ESCAPE_HOLE", "AVOID_WATER", "AVOID_DIGGING_TRAP", "RECOVER_CAMERA"]
    labels = ["Exit water", "Reacquire stone", "Climb shore", "Escape hole", "Avoid water", "Avoid trap", "Recover camera"]
    baseline = [micro["tasks"][name]["success_rate"] * 100 for name in order]
    candidate = [ppo["micro_benchmarks"][name]["candidate_success_rate"] * 100 for name in order]
    positions = range(len(order))
    fig, axis = plt.subplots(figsize=(10.5, 5.0))
    width = 0.36
    axis.bar([x - width / 2 for x in positions], baseline, width, label="r32/6400", color="#2B7A78")
    axis.bar([x + width / 2 for x in positions], candidate, width, label="PPO candidate", color="#D98D58")
    axis.set_xticks(list(positions), labels, rotation=25, ha="right")
    axis.set_ylim(0, 105)
    axis.set_ylabel("Verified success rate (%)")
    axis.set_title("Controlled recovery tests exposed near-zero capabilities; PPO did not repair them")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def save_iron_chart():
    path = FIGURES / "iron_milestones.png"
    seeds, milestones, rates = common_milestone_rates()
    labels = ["Logs", "Wood pick", "3 cobble", "Stone pick", "Furnace", "3 raw iron", "3 ingots", "Iron pick"]
    fig, axis = plt.subplots(figsize=(10.5, 5.0))
    styles = [("Full", "#164E70", "o"), ("No specialist", "#D97732", "s"), ("No System 2", "#6B7280", "^")]
    for arm, color, marker in styles:
        values = [value * 100 for value in rates[arm]]
        axis.plot(labels, values, label=arm, color=color, marker=marker, linewidth=2.2, markersize=6)
    axis.set_ylim(-4, 104)
    axis.set_ylabel("Milestone completion (%)")
    axis.set_title(f"Long-horizon composition failed before smelting (paired common seeds, n={len(seeds)})")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def save_snapshots():
    path = FIGURES / "milestone_snapshots.png"
    images = [
        (FIGURES / "no_system2_wooden_pickaxe.jpg", "Scripted hierarchy: wooden pickaxe"),
        (FIGURES / "milestone_stone_pickaxe.jpg", "Full stack: stone pickaxe milestone"),
        (FIGURES / "milestone_raw_iron_3.jpg", "Full stack: rare 3-raw-iron milestone"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for axis, (image_path, title) in zip(axes, images):
        axis.imshow(PILImage.open(image_path))
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle("Representative verifier-triggered snapshots", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def page_decor(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2E8"))
    canvas.line(MARGIN_LEFT, PAGE_HEIGHT - 1.1 * cm, PAGE_WIDTH - MARGIN_RIGHT, PAGE_HEIGHT - 1.1 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN_LEFT, 0.8 * cm, "System 0/1/2 Minecraft Agent Study")
    page_text = str(document.page)
    canvas.drawRightString(PAGE_WIDTH - MARGIN_RIGHT, 0.8 * cm, page_text)
    canvas.restoreState()


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PaperTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25, leading=29, textColor=INK, alignment=TA_LEFT, spaceAfter=14))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=12.5, leading=17, textColor=BLUE, spaceAfter=18))
    styles.add(ParagraphStyle(name="Abstract", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=INK, alignment=TA_JUSTIFY, backColor=LIGHT, borderPadding=10, spaceAfter=14))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=BLUE, spaceBefore=13, spaceAfter=7))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=INK, spaceBefore=10, spaceAfter=5))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.4, leading=13.3, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7))
    styles.add(ParagraphStyle(name="Bulletx", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, leftIndent=13, firstLineIndent=-7, bulletIndent=2, textColor=INK, spaceAfter=4))
    styles.add(ParagraphStyle(name="Captionx", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.1, leading=10.8, textColor=MUTED, alignment=TA_CENTER, spaceBefore=3, spaceAfter=9))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=colors.white, backColor=BLUE, borderPadding=9, spaceBefore=6, spaceAfter=10))
    styles.add(ParagraphStyle(name="Reference", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.2, leading=11, textColor=INK, leftIndent=12, firstLineIndent=-12, spaceAfter=4))
    return styles


def p(text, styles, style="Bodyx"):
    return Paragraph(text, styles[style])


def figure(path, caption, styles, width=16.6 * cm):
    with PILImage.open(path) as source:
        pixel_width, pixel_height = source.size
    height = width * pixel_height / pixel_width
    image = Image(str(path), width=width, height=height)
    return KeepTogether([image, p(caption, styles, "Captionx")])


def result_table(data, widths, styles):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("LEADING", (0, 0), (-1, -1), 10.5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D3DA")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def build():
    behavioral = load_json("behavioral_screen_report.json")
    isolation = load_json("system0_isolation_report.json")
    micro = load_json("system1_recovery_micro_report.json")
    ppo = load_json("ppo_final_report.json")
    architecture = load_json("architecture_report.json")
    stage0 = load_json("stage0_baseline_summary.json")
    iron_full = load_json("iron_full_summary.json")
    iron_no_specialist = load_json("iron_no_specialist_summary.json")
    iron_no_system2 = load_json("iron_no_system2_summary.json")
    save_architecture()
    save_behavioral_chart(behavioral)
    save_boundary_chart(isolation)
    save_micro_chart(micro, ppo)
    save_iron_chart()
    save_snapshots()
    styles = make_styles()
    document = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=MARGIN_LEFT,
        rightMargin=MARGIN_RIGHT,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
        title="Where Modular Minecraft Agents Break",
        author="Independent MineStudio research project",
        subject="Empirical study of a System 2, System 1, and System 0 Minecraft agent architecture",
    )
    frame = Frame(MARGIN_LEFT, MARGIN_BOTTOM, PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT, PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM, id="paper")
    document.addPageTemplates([PageTemplate(id="paper", frames=[frame], onPage=page_decor)])
    story = []
    story.append(Spacer(1, 1.1 * cm))
    story.append(p("Where Modular Minecraft Agents Break", styles, "PaperTitle"))
    story.append(p("A controlled study of deterministic mechanics, learned visuomotor specialists, and frontier-model planning in MineStudio", styles, "Subtitle"))
    story.append(p("Independent MineStudio research report | September 2026", styles, "H2x"))
    story.append(Spacer(1, 0.4 * cm))
    abstract = (
        "<b>Abstract.</b> We evaluated a three-level Minecraft agent in which a frontier multimodal model performs planning (System 2), "
        "verified programs execute deterministic mechanics (System 0), and STEVE-1-derived policies control uncertain visual interaction (System 1). "
        "The decomposition yielded a real but narrow result: targeted behavior cloning increased paired stone-acquisition success from 37% to 66% "
        "(+29 percentage points; paired bootstrap 95% CI +17 to +41; exact McNemar p=8.96e-6). However, seemingly safer System 0 handoffs reduced success "
        "from 66% to 57%, and targeted PPO reduced the champion from 66% to 54%. In a fresh-world iron-pickaxe study, none of the three systems completed "
        "the objective. The full architecture reached a stone pickaxe in 15/22 valid runs and three raw iron in 2/22, but produced no ingots; vanilla System 1 "
        "with the same planner reached a stone pickaxe in 18/24 and raw iron in 1/24. The scripted no-System-2 arm never reached a stone pickaxe. These results "
        "support modular decomposition as a diagnostic and engineering tool, but do not support the stronger claim that the implemented architecture solves long-horizon Minecraft."
    )
    story.append(p(abstract, styles, "Abstract"))
    story.append(p("Bottom line", styles, "H1x"))
    story.append(p("The project found a publishable negative result: <b>local competence can improve substantially without composing into long-horizon autonomy.</b> The abstraction boundaries themselves became failure surfaces. Better imitation loss did not reliably imply better behavior, removing deterministic errors did not reliably improve end-to-end success, and verifier-driven PPO did not preserve the strongest behavioral checkpoint.", styles, "Callout"))
    story.append(figure(FIGURES / "architecture.png", "Figure 1. System 2 chooses and revises objectives; the router dispatches exact mechanics to System 0 and uncertain visual control to System 1; MineStudio supplies pixels, actions, events, and exact verification.", styles))
    story.append(PageBreak())

    story.append(p("1. Motivation and research question", styles, "H1x"))
    story.append(p("Minecraft combines long-horizon planning, partially observed navigation, continuous camera control, discrete inventory interfaces, and exact crafting dependencies. VPT demonstrated that large-scale video pretraining can produce a broad behavioral prior from native mouse and keyboard actions [1]. STEVE-1 added open-ended text conditioning to that prior [2]. MineStudio made these models, simulation, fine-tuning, and benchmarking accessible under a common environment API [3].", styles))
    story.append(p("The project tested a specific systems hypothesis: high-level reasoning should be rented from an interchangeable frontier model; open-world motor competence should be inherited and selectively adapted; deterministic mechanics should be executed by verified programs. This resembles classical hierarchical control, but the practical question is empirical: <b>does the decomposition improve complete tasks, or merely move errors between modules?</b>", styles))
    story.append(p("Our contributions are:", styles, "H2x"))
    for item in [
        "A behavioral, paired selection protocol showing that the offline-loss winner was not the Minecraft-behavior winner.",
        "A System 0 library with explicit preconditions, success predicates, timeouts, aborts, and structured handoffs.",
        "Seven randomized recovery micro-benchmarks that separated an almost-solved skill (exit water) from near-zero skills (camera recovery and trap avoidance).",
        "A targeted PPO result showing modest development gains but statistically harmful end-to-end hazard performance.",
        "A long-horizon iron-pickaxe viability test that falsified the strong version of the architecture thesis in its current implementation.",
    ]:
        story.append(p(item, styles, "Bulletx"))

    story.append(p("2. Architecture", styles, "H1x"))
    story.append(p("<b>System 2</b> received sampled frames and structured state, maintained the global objective, emitted a short-term objective and immediate instruction, interpreted failures, and replanned. In the final study it was Gemini 3.8 Flash through a structured JSON interface. No hidden map, resource coordinates, or world-state oracle was exposed.", styles))
    story.append(p("<b>System 1</b> was the native-pixel, native-action controller. Vanilla STEVE-1 handled general open-world behavior. The selected specialist was an upper-layer LoRA with rank 32 trained for stone acquisition and recovery. Routing used the specialist only for relevant stone/recovery instructions and vanilla STEVE-1 elsewhere.", styles))
    story.append(p("<b>System 0</b> contained bounded action packets for crafting, equipment, placement, station lifecycle, mining completion, pickup, and related mechanics. Every skill exposed preconditions, execution, success and failure predicates, timeout, abort conditions, and a structured result including <font name='Courier'>NEEDS_SYSTEM1</font> when geometry became uncertain. Six initial skills passed live smoke tests and the contract/adapter suite passed 17 tests.", styles))
    story.append(p("This boundary deliberately excluded open-ended search and hidden-map pathfinding. System 0 answered how to execute a known operation; System 1 handled uncertain perception and terrain; System 2 decided what to do and why.", styles))

    story.append(p("3. Experimental design", styles, "H1x"))
    experiment_rows = [
        ["Experiment", "Episodes", "Question", "Selection rule"],
        ["Stage-0 stone baseline", "1,000", "Where does pretrained execution fail?", "Exact inventory/event labels"],
        ["Behavioral policy screen", "800", "Which BC checkpoint acts best?", "Paired 40 normal + 60 hazard per policy"],
        ["System 0/router isolation", "400", "Which boundary change caused regression?", "Four paired configurations"],
        ["Recovery micro-benchmarks", "350", "Which System 1 skills are weakest?", "50 episodes x 7 exact verifiers"],
        ["Targeted PPO", "Training + 450 eval", "Can RL repair residual skills?", "Behavior plus preserved stone competence"],
        ["Iron-pickaxe viability", "70 valid of 72", "Does the full stack compose?", "Paired milestones; no training"],
    ]
    story.append(result_table(experiment_rows, [3.2 * cm, 2.2 * cm, 6.4 * cm, 4.9 * cm], styles))
    story.append(p("All direct policy comparisons used identical seeds and initial states. Success was defined from MineStudio state, not from language-model judgment. Gemini critiques were diagnostic labels, never numerical reward. The final long-horizon run stopped at 70 valid episodes because of external API spending limits; the full arm contains 22/24 valid seeds, while both controls contain 24/24. We report this incompleteness explicitly and make no claim of final-success superiority.", styles))

    story.append(p("4. Results", styles, "H1x"))
    story.append(p("4.1 Baseline success hid a concentrated hazard failure", styles, "H2x"))
    stage0_overall = stage0["overall"]
    water = stage0["by_scenario"]["water_near_stone"]
    swamp = stage0["by_biome"]["swamp"]
    story.append(p(f"Across the broad 1,000-episode baseline distribution, the current STEVE-plus-options system acquired three cobblestone in <b>{stage0_overall['successes']}/1,000 ({stage0_overall['success_rate']*100:.1f}%)</b>, with a Wilson 95% interval of {stage0_overall['success_wilson_95'][0]*100:.1f}% to {stage0_overall['success_wilson_95'][1]*100:.1f}%. Aggregate success was misleading: water-near-stone success was only <b>{water['success_rate']*100:.1f}%</b>, and swamp success was <b>{swamp['success_rate']*100:.1f}%</b>. Primary failures included 86 water-entry failures and 43 block-broken-without-collection failures.", styles))
    story.append(p("This established a useful architectural distinction. Pickup completion was a deterministic tail suitable for System 0; losing the target, terrain traps, and water recovery were learned-control problems. The remainder of the study tested whether that diagnosis led to reliable improvements.", styles))

    story.append(p("4.2 Targeted BC produced a strong narrow specialist", styles, "H2x"))
    story.append(figure(FIGURES / "behavioral_screen.png", "Figure 2. Paired stone-acquisition screen. Rank-32 upper LoRA at step 6400 achieved 66% combined success versus 37% for vanilla, despite rank-8 having the best offline validation loss.", styles))
    winner = next(row for row in behavioral["policies"] if row["policy"] == "upper_lora_r32_step6400")
    offline_winner = architecture["winner"]
    story.append(p(f"The offline sweep selected <b>{offline_winner}</b> by validation loss (2.062), but the behavioral winner was rank-32 upper LoRA. It achieved 75% normal and 60% hazard success, versus 47.5% and 30% for vanilla. The combined +29-point effect had paired bootstrap 95% CI +17 to +41 points and exact McNemar p={winner['versus_vanilla']['mcnemar']['exact_p_value']:.2e}. Water-conditioned success improved from 27.9% to 58.2%, and post-water stone reacquisition improved from 32.4% to 67.2%.", styles))
    story.append(p("This is the clearest positive result: modest parameter-efficient adaptation of a pretrained motor prior can double performance in a targeted hazard distribution. It also demonstrates why behavioral checkpoint selection is mandatory; continuing the lower-loss rank-8 run to 6400 steps reduced hazard success from 53.3% at step 4800 to 35.0%.", styles))

    story.append(p("4.3 System 0 correctness was not sufficient", styles, "H2x"))
    story.append(figure(FIGURES / "system0_boundary.png", "Figure 3. Hardened pickup eliminated strict pickup-tail failures but reduced end-to-end success. Router changes also failed to improve the champion.", styles))
    story.append(p("The old router and old pickup controller remained the 66% champion. Hardened pickup with the old router eliminated strict pickup-tail failures (3 to 0) but reduced success to 58%. The new router with old pickup scored 62%; combining both changes scored 57%. The hardened controller returned <font name='Courier'>NEEDS_SYSTEM1</font> 36 times, but the handoff did not reliably restore productive behavior. A locally safer contract therefore created a worse global control loop.", styles))
    story.append(p("This result is important beyond Minecraft: modular systems cannot be optimized independently when handoff semantics change the state distribution seen by downstream controllers. Verification prevented false local success, but it did not guarantee useful recovery.", styles))

    story.append(p("4.4 Micro-benchmarks localized the real System 1 deficit", styles, "H2x"))
    story.append(figure(FIGURES / "micro_and_ppo.png", "Figure 4. Exact controlled recovery success for the immutable r32/6400 specialist and the best PPO candidate. Exit-water was nearly solved; camera recovery and trap avoidance remained at zero.", styles))
    micro_rows = [["Capability", "r32/6400", "PPO candidate", "Dominant failure"]]
    task_labels = {
        "EXIT_WATER": "Exit water",
        "REACQUIRE_STONE": "Reacquire stone",
        "CLIMB_SHORE": "Climb shore",
        "ESCAPE_HOLE": "Escape hole",
        "AVOID_WATER": "Avoid water",
        "AVOID_DIGGING_TRAP": "Avoid digging trap",
        "RECOVER_CAMERA": "Recover camera",
    }
    failure_labels = {
        "EXIT_WATER": "still in water",
        "REACQUIRE_STONE": "stone not reacquired",
        "CLIMB_SHORE": "failed shore climb",
        "ESCAPE_HOLE": "remained in hole",
        "AVOID_WATER": "entered water / target lost",
        "AVOID_DIGGING_TRAP": "dug into trap",
        "RECOVER_CAMERA": "camera not recovered",
    }
    for name in ["EXIT_WATER", "REACQUIRE_STONE", "CLIMB_SHORE", "ESCAPE_HOLE", "AVOID_WATER", "AVOID_DIGGING_TRAP", "RECOVER_CAMERA"]:
        micro_rows.append([task_labels[name], f"{micro['tasks'][name]['success_rate']*100:.0f}%", f"{ppo['micro_benchmarks'][name]['candidate_success_rate']*100:.0f}%", failure_labels[name]])
    story.append(result_table(micro_rows, [4.2 * cm, 2.7 * cm, 3.2 * cm, 6.6 * cm], styles))
    story.append(p("The asymmetry was sharp. Exiting water succeeded in 49/50 episodes, but climbing a shore succeeded in 5/50, escaping a hole in 4/50, avoiding water in 2/50, and both trap avoidance and camera recovery in 0/50. Gemini criticized all 50 camera failures as bad camera orientation and all 50 trap failures as mining into a trap; exact environment predicates independently confirmed the failures.", styles))

    story.append(p("4.5 Targeted PPO failed the promotion gate", styles, "H2x"))
    story.append(figure(FIGURES / "ppo_training_curves.png", "Figure 5. PPO diagnostics. Development reacquisition briefly improved, but most weak tasks remained flat and final end-to-end hazard performance regressed.", styles))
    combined = ppo["stone"]["combined"]
    hazard = ppo["stone"]["hazard"]
    story.append(p(f"The best development checkpoint improved the weak-task mean from 13.3% to 20.0% and normal-stone success from 70% to 75%. That signal did not survive the frozen final evaluation. Reacquisition moved from 30% to 34%, camera recovery and trap avoidance remained at 0%, and exit-water fell from 98% to 92%. End-to-end stone success fell from <b>{combined['baseline_success_rate']*100:.0f}% to {combined['candidate_success_rate']*100:.0f}%</b> (difference {combined['difference']*100:.0f} points; paired bootstrap 95% CI {combined['paired_bootstrap_95_ci'][0]*100:.0f} to {combined['paired_bootstrap_95_ci'][1]*100:.0f}). Hazard success fell 18.3 points with exact McNemar p={hazard['mcnemar']['exact_p_value']:.3f}. The PPO candidate was correctly rejected.", styles))

    story.append(p("4.6 Long-horizon viability: no system built an iron pickaxe", styles, "H2x"))
    story.append(figure(FIGURES / "iron_milestones.png", "Figure 6. Milestone completion on the 22 seeds common to all three arms. The full architecture and no-specialist control progressed farther than the fixed hierarchy, but neither reached smelted iron.", styles))
    story.append(figure(FIGURES / "milestone_snapshots.png", "Figure 7. Verifier-triggered gameplay snapshots. The architecture repeatedly established early tools and occasionally reached raw iron, but never produced three iron ingots.", styles))
    long_rows = [
        ["Arm", "Valid", "Stone pickaxe", "Furnace", "3 raw iron", "3 ingots", "Iron pickaxe"],
        ["Full System", "22", f"{iron_full['milestones']['stone_pickaxe']['achieved']}/22", f"{iron_full['milestones']['furnace_available']['achieved']}/22", f"{iron_full['milestones']['raw_iron_3']['achieved']}/22", "0/22", "0/22"],
        ["No specialist", "24", f"{iron_no_specialist['milestones']['stone_pickaxe']['achieved']}/24", f"{iron_no_specialist['milestones']['furnace_available']['achieved']}/24", f"{iron_no_specialist['milestones']['raw_iron_3']['achieved']}/24", "0/24", "0/24"],
        ["No System 2", "24", "0/24", "0/24", "0/24", "0/24", "0/24"],
    ]
    story.append(result_table(long_rows, [3.4 * cm, 1.5 * cm, 2.6 * cm, 2.3 * cm, 2.4 * cm, 2.2 * cm, 2.5 * cm], styles))
    story.append(p("The scripted no-System-2 controller obtained logs in all 24 runs and three cobblestone in 13, but never produced a stone pickaxe, exposing a deterministic hierarchy/controller defect. Live System 2 clearly moved farther: the two Gemini arms produced stone pickaxes and furnaces. Yet the specialist did not improve composition. On their reported valid denominators, Full reached a stone pickaxe in 68.2% and a furnace in 50.0%, while No Specialist reached 75.0% and 70.8%. Full reached three raw iron twice versus once for No Specialist, but these counts are too small for a superiority claim. No arm produced an iron ingot.", styles))
    story.append(p("The final two Full episodes were not run after prepaid API credit was exhausted. Because no arm had any iron-pickaxe success and all observed collapse occurred before smelting, the missing episodes cannot rescue the central viability claim. They do, however, prevent a formal 24-seed paired comparison of intermediate milestone rates.", styles))

    story.append(p("5. What belongs to System 0, System 1, and System 2?", styles, "H1x"))
    ownership_rows = [
        ["Owner", "Supported responsibility", "Observed residual failure"],
        ["System 0", "Crafting, equipment, station use, known-target tails", "Local correctness did not ensure a recoverable handoff; fixed hierarchy also stalled before stone-pickaxe completion"],
        ["System 1", "Search, approach, terrain, camera, target reacquisition", "0% camera recovery, 0% trap avoidance, 8% hole escape, 10% shore climb"],
        ["System 2", "State tracking, objective choice, replanning", "Advanced beyond the fixed controller, but repeated infeasible crafting and recovery choices; roughly 28-29 replans per long episode"],
        ["Interface/router", "Preserve context and switch at stable boundaries", "Strict ambiguity handoffs and reduced transition counts still degraded success"],
    ]
    story.append(result_table(ownership_rows, [2.7 * cm, 6.1 * cm, 7.9 * cm], styles))
    story.append(p("The evidence does not identify a single failed module. System 1 is the dominant bottleneck for controlled recovery, but the router determines whether that weakness is exposed or amplified. System 2 contributes useful hierarchy traversal, yet cannot compensate for unreliable motor recovery. System 0 removes deterministic variance, but only when its success and failure contracts match what System 1 can resume from.", styles))

    story.append(p("6. Discussion", styles, "H1x"))
    story.append(p("<b>The narrow result is real.</b> A rank-32 LoRA improved a difficult paired benchmark by 29 points with strong statistical evidence. This supports interchangeable task specialists as a practical way to adapt a large motor prior without retraining a foundation policy.", styles))
    story.append(p("<b>The broad thesis is not yet supported.</b> The specialist gain did not improve the long-horizon iron hierarchy, and the full architecture achieved zero final successes. Relative to OpenAI VPT or published STEVE-1 results, this study is neither a model-scale comparison nor a reproduction: tasks, budgets, prompts, checkpoints, and evaluation distributions differ. It would be incorrect to claim superiority over those systems.", styles))
    story.append(p("<b>Modularity created measurable interface debt.</b> The most revealing experiment was not training. Eliminating all strict pickup-tail failures made the agent worse because ambiguity triggered handoffs that the learned controller could not exploit. The correct unit of optimization is therefore a closed loop spanning option termination, state representation, router policy, and resumed controller context.", styles))
    story.append(p("<b>Behavioral selection must dominate proxy losses.</b> Rank-8 won offline; rank-32 won online. PPO reward and development micro-tests also selected a checkpoint that regressed on the paired hazard benchmark. This pattern argues for small, frozen, behaviorally meaningful gates throughout training.", styles))
    story.append(p("<b>LLM spending did not purchase motor competence.</b> Gemini was valuable for planning and failure taxonomy, but repeated calls were expensive and did not solve camera, terrain, or target reacquisition. Exact verifiers were cheaper and more reliable for rewards and completion. An economical future system should invoke System 2 only on meaningful state changes, cache plans, and keep critic sampling sparse.", styles))

    story.append(p("7. Limitations", styles, "H1x"))
    for item in [
        "The study is a single-project engineering investigation, not a standardized MineRL leaderboard submission.",
        "The 1,000-episode stage-0 distribution and the 100-episode behavioral screen differ; their absolute success rates must not be compared directly.",
        "The final Full arm contains 22 rather than 24 valid episodes because external API credit was exhausted.",
        "Gemini critiques are correlated observations, not independent ground truth; environment predicates control the quantitative claims.",
        "The long-horizon controller exposed crafting and fixed-hierarchy defects, so the iron-pickaxe test mixes component quality with integration quality.",
        "No claims are made about human-level play, general Minecraft competence, or superiority to published VPT, STEVE-1, Voyager, or MineStudio baselines.",
    ]:
        story.append(p(item, styles, "Bulletx"))

    story.append(p("8. Conclusion", styles, "H1x"))
    story.append(p("The System 0/1/2 framing was useful, but the current agent is not a viable long-horizon Minecraft system. The project demonstrated that targeted BC can create a materially stronger specialist and that deterministic options can compress exact mechanics. It also demonstrated three negative results: module-local correctness can reduce system success, targeted PPO can erase a good behavioral prior, and live frontier-model planning cannot overcome a weak recovery interface. The defensible conclusion is not that the decomposition failed conceptually; it is that <b>composition must be trained and evaluated as its own capability</b>. Without that, the architecture remains an insightful diagnostic scaffold around an insufficient body.", styles))
    story.append(p("Given zero iron-pickaxe completions and the high API cost of the final evaluation, stopping additional patching was rational. A future restart should require a cheaper deterministic integration harness, explicit option-to-policy recovery state, and a pre-registered promotion gate on complete long-horizon milestones.", styles))

    story.append(p("9. Reproducibility and artifact disposition", styles, "H1x"))
    story.append(p("The retained DGX state contains the official STEVE-1 checkpoint, the immutable rank-32/6400 specialist, compact JSON reports and result rows, four representative images, SHA-256 manifests, and source code. Raw rollouts, smoke outputs, superseded architectures, failed PPO candidates, and unused GROOT/ROCKET/VPT checkpoints were removed after the compact archive and champion were hash-verified. The cleanup released approximately 216 GB from the MineStudio project tree. The paper source bundle contains the same reports used to generate every table and chart.", styles))

    story.append(p("References", styles, "H1x"))
    references = [
        "[1] B. Baker et al. Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos. NeurIPS, 2022. https://arxiv.org/abs/2206.11795",
        "[2] S. Lifshitz, K. Paster, H. Chan, J. Ba, and S. McIlraith. STEVE-1: A Generative Model for Text-to-Behavior in Minecraft. NeurIPS, 2023. https://arxiv.org/abs/2306.00937",
        "[3] S. Cai et al. MineStudio: A Streamlined Package for Minecraft AI Agent Development. arXiv:2412.18293, 2024. https://arxiv.org/abs/2412.18293",
        "[4] G. Wang et al. Voyager: An Open-Ended Embodied Agent with Large Language Models. TMLR, 2024. https://arxiv.org/abs/2305.16291",
        "[5] J. Schulman et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347",
        "[6] E. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022. https://arxiv.org/abs/2106.09685",
    ]
    for reference in references:
        story.append(p(reference, styles, "Reference"))
    document.build(story)


build()
