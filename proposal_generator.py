#!/usr/bin/env python3
"""
Pivot Proposal Generator
Generates structured, compelling proposals for tech, healthcare, and social impact projects.
"""

import sys
import anthropic


SYSTEM_PROMPT = """You are Pivot's expert proposal writer — a specialist in crafting compelling,
structured proposals for technology, healthcare, and social impact projects.

When generating a proposal, always structure it with these sections:
1. Executive Summary
2. Problem Statement
3. Proposed Solution
4. Key Features & Capabilities
5. Expected Impact & Benefits
6. Implementation Plan (with phases/timeline)
7. Budget Overview (high-level estimates)
8. Team & Qualifications
9. Conclusion & Call to Action

Write with clarity and conviction. Tailor the tone to the audience (investors, grant committees,
clients, or internal stakeholders). Back claims with logical reasoning. Keep language accessible
yet professional. Proposals should be thorough but concise — aim for quality over length."""


def gather_project_info() -> dict:
    """Interactively gather information about the project."""
    print("\n" + "=" * 60)
    print("  PIVOT PROPOSAL GENERATOR")
    print("=" * 60)
    print("\nAnswer a few questions to generate your proposal.\n")

    fields = [
        ("project_name", "Project / Initiative name"),
        ("organization", "Your organization / team name"),
        ("audience", "Who is this proposal for? (e.g., investors, grant committee, client)"),
        ("problem", "What problem are you solving? (describe in a few sentences)"),
        ("solution", "What is your proposed solution?"),
        ("impact", "Who benefits and how? What is the expected impact?"),
        ("timeline", "What is your estimated timeline? (e.g., 12 months)"),
        ("budget", "What is your estimated budget or funding ask? (e.g., $500K)"),
        ("team", "Briefly describe your team or key qualifications"),
        ("extra", "Any additional context or requirements? (press Enter to skip)"),
    ]

    info = {}
    for key, label in fields:
        value = input(f"{label}: ").strip()
        if value:
            info[key] = value

    return info


def build_prompt(info: dict) -> str:
    """Build the user prompt from collected project information."""
    lines = [
        "Please generate a comprehensive, professional proposal based on the following information:\n"
    ]

    field_labels = {
        "project_name": "Project Name",
        "organization": "Organization",
        "audience": "Target Audience",
        "problem": "Problem Statement",
        "solution": "Proposed Solution",
        "impact": "Expected Impact",
        "timeline": "Timeline",
        "budget": "Budget / Funding Ask",
        "team": "Team & Qualifications",
        "extra": "Additional Context",
    }

    for key, label in field_labels.items():
        if key in info and info[key]:
            lines.append(f"**{label}:** {info[key]}")

    lines.append(
        "\nGenerate a polished, well-structured proposal. "
        "Use clear headings, bullet points where appropriate, and persuasive language. "
        "Make it compelling and tailored to the specified audience."
    )

    return "\n".join(lines)


def generate_proposal(info: dict) -> None:
    """Generate the proposal using Claude API with streaming."""
    client = anthropic.Anthropic()

    prompt = build_prompt(info)

    print("\n" + "=" * 60)
    print("  GENERATING PROPOSAL...")
    print("=" * 60 + "\n")

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=64000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        in_thinking = False
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "thinking":
                    in_thinking = True
                    print("[Thinking...]\n", flush=True)
                elif event.content_block.type == "text":
                    if in_thinking:
                        in_thinking = False
                        print("\n" + "-" * 60 + "\n", flush=True)
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)

    final = stream.get_final_message()
    usage = final.usage
    print("\n\n" + "=" * 60)
    print(f"  Proposal generated successfully.")
    print(f"  Tokens used — Input: {usage.input_tokens:,}  |  Output: {usage.output_tokens:,}")
    print("=" * 60 + "\n")


def save_proposal_option(info: dict) -> None:
    """Optionally save the proposal to a file."""
    save = input("Save proposal to a file? (y/n): ").strip().lower()
    if save != "y":
        return

    project_slug = (
        info.get("project_name", "proposal")
        .lower()
        .replace(" ", "_")
        .replace("/", "_")[:40]
    )
    filename = f"{project_slug}_proposal.md"

    # Re-generate to file
    client = anthropic.Anthropic()
    prompt = build_prompt(info)

    print(f"\nSaving to {filename}...")

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=64000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        content = ""
        for event in stream:
            if (
                event.type == "content_block_delta"
                and event.delta.type == "text_delta"
            ):
                content += event.delta.text

    header = f"# {info.get('project_name', 'Proposal')}\n\n"
    if info.get("organization"):
        header += f"**Organization:** {info['organization']}\n\n"

    with open(filename, "w") as f:
        f.write(header + content)

    print(f"Proposal saved to: {filename}\n")


def main() -> None:
    try:
        info = gather_project_info()

        if not any(info.get(k) for k in ("problem", "solution", "project_name")):
            print("\nError: Please provide at least a project name, problem, and solution.")
            sys.exit(1)

        generate_proposal(info)
        save_proposal_option(info)

    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)
    except anthropic.AuthenticationError:
        print("\nError: Invalid API key. Set the ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)
    except anthropic.APIConnectionError:
        print("\nError: Could not connect to the Anthropic API. Check your internet connection.")
        sys.exit(1)
    except anthropic.RateLimitError:
        print("\nError: Rate limit reached. Please wait a moment and try again.")
        sys.exit(1)
    except anthropic.APIStatusError as e:
        print(f"\nAPI error ({e.status_code}): {e.message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
