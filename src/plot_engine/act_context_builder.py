# src/plot_engine/act_context_builder.py

from typing import Any, Dict, List


class ActContextBuilder:
    """Builds rich, structured context for act planning"""
    def build_act_planner_context(
        self,
        story_data: Dict[str, Any],
        act_number: int,
        story_mode: str  # "compact", "standard", "epic"
    ) -> Dict[str, Any]:
        """
        Build optimized context for act planner based on story mode.
        """
        
        base_context = {
            "seed": story_data["seed"],
            "final_plot": story_data["final_plot"],
            "connected_agents": story_data["connected_agents"],
            "integrated_world": story_data["integrated_world"],
            "conflict_matrix": story_data["conflict_matrix"],
            "story_tracker": story_data["story_tracker"],
        }
        
        if story_mode == "compact":
            # Compact mode - base context is enough
            return base_context
        
        elif story_mode == "standard":
            # Standard mode - base context is enough
            return base_context
        
        elif story_mode == "epic":
            # Epic mode - add summarized enhancements
            backstories = story_data.get("character_backstories", [])
            world_guide = story_data.get("world_guide")
            subplot_arch = story_data.get("subplot_architecture")
            
            # Create backstory highlights (not full text)
            backstory_highlights = []
            for bs in backstories:
                backstory_highlights.append({
                    "name": bs.name,
                    "key_formative_events": bs.formative_events[:300] + "...",  # First 300 chars
                    "psychological_core": bs.psychological_profile[:200] + "...",
                    "secrets": bs.secrets[:3],  # Top 3 secrets
                    "subplot_seeds": bs.subplot_seeds[:2],  # Top 2 subplot ideas
                })

            base_context.update({
                "backstory_highlights": backstory_highlights,
                "key_locations": world_guide.location_dossiers if world_guide else [],
                "npc_pool": world_guide.minor_character_pool if world_guide else [],
                "subplot_architecture": subplot_arch,
                "subplot_integration_for_act": self._extract_subplot_beats_for_act(
                    subplot_arch, act_number
                ) if subplot_arch else []
            })
            
            return base_context
        
        return base_context


    def _extract_subplot_beats_for_act(
        self,
        subplot_arch: Any,
        act_number: int
    ) -> List[Dict[str, str]]:
        """Extract only the subplot beats relevant to this act"""
        beats = []
        
        for subplot in subplot_arch.subplots:
            if act_number in subplot.act_integration:
                beats.append({
                    "subplot_title": subplot.title,
                    "character_owner": subplot.character_owner,
                    "beat_this_act": subplot.act_integration[act_number],
                    "thematic_connection": subplot.thematic_connection
                })
        return beats

    

    @staticmethod
    def build_act_context(
        act_number: int,
        final_plot: dict,
        seed: dict,
        connected_agents: list,
        integrated_world: dict,
        conflict_matrix: dict,
        story_tracker: dict,
        previous_acts: dict,
        story_so_far: str,
        word_guidance: tuple
    ) -> str:
        """Build comprehensive act context"""
        
        word_percentage, act_guidance = word_guidance
        suggested_words = int(seed.get('target_length', 50000) * word_percentage)
        suggested_chapters = max(2, suggested_words // 2500)
        
        context_parts = []
        
        context_parts.append(f"""
╔═══════════════════════════════════════════════════════════════════════════════
║ ACT {act_number} CONTEXT & GUIDANCE
╚═══════════════════════════════════════════════════════════════════════════════

ACT PURPOSE: {act_guidance}
TARGET: ~{suggested_words:,} words across {suggested_chapters} chapters
TONE: {seed.get('tone', 'Balanced')}
PROSE STYLE: {seed.get('prose_style', 'Standard narrative')}
""")
        
        act_summaries = final_plot.get('act_summaries', [])
        if act_number <= len(act_summaries):
            context_parts.append(f"""
┌─ ACT SUMMARY ────────────────────────────────────────────────────────────────
{act_summaries[act_number - 1]}
└──────────────────────────────────────────────────────────────────────────────
""")
        
        outstanding_setup = []
        for prev_act_num in range(1, act_number):
            prev_tracking = story_tracker.get(str(prev_act_num), {})
            if isinstance(prev_tracking, dict):
                setup = prev_tracking.get('setup_elements', [])
                outstanding_setup.extend([f"(Act {prev_act_num}) {item}" for item in setup])
        
        current_tracking = story_tracker.get(str(act_number), {})
        if isinstance(current_tracking, dict):
            expected_payoffs = current_tracking.get('payoff_elements', [])
            expected_setups = current_tracking.get('setup_elements', [])
        else:
            expected_payoffs = []
            expected_setups = []
        
        context_parts.append(f"""
┌─ STORY CONTINUITY ───────────────────────────────────────────────────────────

OUTSTANDING SETUP (needs payoff in this or future acts):
{ActContextBuilder._format_list(outstanding_setup[:10], indent=2) if outstanding_setup else "  • (none - this is Act 1)"}

EXPECTED PAYOFFS THIS ACT:
{ActContextBuilder._format_list(expected_payoffs, indent=2)}

EXPECTED SETUP THIS ACT (for future payoff):
{ActContextBuilder._format_list(expected_setups, indent=2)}
└──────────────────────────────────────────────────────────────────────────────
""")
        
        context_parts.append("\n┌─ AGENT ARCS THIS ACT ────────────────────────────────────────────────────────")
        
        tracker_agents = {}
        if isinstance(current_tracking, dict):
            tracker_agents = current_tracking.get('key_agent_moments', {})
        
        for agent in connected_agents:
            agent_name = agent.get('name', 'Unknown')
            agent_role = agent.get('role', '')
            agent_arc = agent.get('agent_arc', '')
            
            moment = tracker_agents.get(agent_name, "")
            
            context_parts.append(f"""
{agent_name} ({agent_role})
  Arc: {agent_arc}
  This Act: {moment if moment else 'Key player in unfolding conflicts'}
""")
        
        context_parts.append("└──────────────────────────────────────────────────────────────────────────────\n")
        
        context_parts.append(f"""
┌─ CONFLICT ESCALATION ────────────────────────────────────────────────────────
Central: {conflict_matrix.get('central_conflict', '')}

Escalation: {conflict_matrix.get('escalation_path', '')}
└──────────────────────────────────────────────────────────────────────────────
""")
        
        world_summary = integrated_world.get('foundation', integrated_world.get('core_concept', ''))
        context_parts.append(f"""
┌─ WORLD & SETTING ────────────────────────────────────────────────────────────
{world_summary}
└──────────────────────────────────────────────────────────────────────────────
""")
        
        if previous_acts and act_number > 1:
            context_parts.append(f"""
┌─ CONSEQUENCES FROM ACT {act_number - 1} ─────────────────────────────────────
{ActContextBuilder._summarize_previous_act(previous_acts.get(act_number - 1, {}))}
└──────────────────────────────────────────────────────────────────────────────
""")
        
        context_parts.append(f"""
┌─ STORY SO FAR ───────────────────────────────────────────────────────────────
{story_so_far}
└──────────────────────────────────────────────────────────────────────────────
""")
        
        themes = seed.get('themes', [])
        if themes:
            narrative_flow = final_plot.get('narrative_flow', '')
            context_parts.append(f"""
┌─ THEMATIC FOCUS ─────────────────────────────────────────────────────────────
Core Themes: {', '.join(themes)}

Development:
{narrative_flow[:500] + '...' if len(narrative_flow) > 500 else narrative_flow}
└──────────────────────────────────────────────────────────────────────────────
""")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def _format_list(items: list, indent: int = 0) -> str:
        """Format list items with indentation"""
        if not items:
            return " " * indent + "(none specified)"
        
        formatted = []
        for item in items:
            if isinstance(item, str):
                clean = item.strip()
                if clean:
                    formatted.append(" " * indent + f"• {clean}")
        
        return "\n".join(formatted) if formatted else " " * indent + "(none specified)"
    
    @staticmethod
    def _summarize_previous_act(prev_act: dict) -> str:
        """Extract key consequences from previous act"""
        climax = prev_act.get('act_climax', '')
        payoffs = prev_act.get('payoff_this_act', [])
        changes = prev_act.get('agent_state_changes', {})
        
        summary = f"Climax: {climax}\n"
        if payoffs:
            summary += f"Payoffs: {', '.join(payoffs[:3])}\n"
        
        if changes:
            summary += "Character Changes: " + ", ".join(
                [f"{agent} ({status})" for agent, status in list(changes.items())[:3]]
            )
        
        return summary
