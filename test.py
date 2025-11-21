# def _get_chapter_outline_from_act(self, chapter_id: int, ctx: StoryContextSnapshot) -> Optional[Dict[str, Any]]:
    #     """Extract chapter outline from act plan"""
    #     chapters = ctx.act_plan.get('chapter_outlines', [])
    #     offset = chapter_id - 1
    #     if offset < len(chapters):
    #         return chapters[offset]
    #     return None

    # def _calculate_chapter_word_target(self, seed: Dict, act_plan: Dict, chapter_id: int) -> int:
    #     """Calculate word target for chapter"""
    #     total_words = seed.get('target_length', 50000)
    #     num_acts = len(self.story_context.final_plot.get('act_summaries', [3]))
    #     chapters_in_act = len(act_plan.get('chapter_outlines', []))
        
    #     return max(2000, total_words // num_acts // chapters_in_act)
    
#     async def chapter_planner_node(self, state: StoryState) -> Dict:
#         print(f"📋 Chapter Planner: Chapter {state.current_chapter_id}")
        
#         # Check if chapter plan already exists
#         existing_plan = await self.memory.get_long_term_document(
#             metadata={
#                 "type": "chapter_plan",
#                 "act_id": state.current_act_id,
#                 "chapter_id": state.current_chapter_id,
#                 "story_title": state.story_title
#             }
#         )
#         if existing_plan:
#             state.chapter_plan = json.loads(existing_plan)
#             print(f"✓ Using existing chapter plan")
#             state.next_action = "scene_planner"
#             return state.__dict__
        
#         # Get chapter outline from act
#         self.current_chapter_outline = self._get_chapter_outline_from_act(
#             state.current_chapter_id,
#             self.story_context
#         )
        
#         if not self.current_chapter_outline:
#             state.error_message = f"Chapter outline not found for chapter {state.current_chapter_id}"
#             state.next_action = "ERROR"
#             return state.__dict__
        
#         # NEW: Format rich context for chapter planning
#         context_formatted = ContextFormatter.full_context(self.story_context)
        
#         target_word_count = self._calculate_chapter_word_target(
#             self.story_context.story_seed,
#             self.story_context.act_plan,
#             state.current_chapter_id
#         )
        
#         system_prompt = f"""You are the Chapter Planner...
# {context_formatted}

# Story Style:
# - POV: {self.story_context.story_seed.get('pov', 'Third-person')}
# - Prose: {self.story_context.story_seed.get('prose_style', '')}
# - Tone: {self.story_context.story_seed.get('tone', '')}

# Chapter Target: ~{target_word_count} words

# {chapter_outline_parser.get_format_instructions()}"""
    
#         human_prompt = f"""CHAPTER FROM AUTHOR:
# {json.dumps(self.current_chapter_outline, indent=2)}

# Create a structured Chapter Outline that guides scene creation."""
        

#         resp, tokens = await director_client(
#             system_prompt=system_prompt,
#             human_prompt=human_prompt,
#             llm_temp=self.llm_temp
#         )
#         self.director_token_usage["prompt_tokens"] += tokens["prompt_tokens"]
#         self.director_token_usage["completion_tokens"] += tokens["completion_tokens"]
#         self.director_token_usage["total_tokens"] += tokens["total_tokens"]

#         clean_resp = StoryHelpers._extract_content(resp)
#         clean_resp = StoryHelpers._strip_code_fences(clean_resp)

#         result = await StoryHelpers.load_json_with_retry(clean_resp, chapter_outline_parser)

#         chapter_plan: ChapterOutline = result
#         chapter_plan.target_word_count = target_word_count

#         # Save chapter plan
#         await self.memory.add_long_term_document(
#             text=chapter_plan.model_dump_json(indent=2),   
#             metadata={
#                 "type": "chapter_plan",
#                 "act_id": state.current_act_id,
#                 "chapter_id": state.current_chapter_id,
#                 "story_title": state.story_title
#             }
#         )
#         await self.memory.update_story_progress({
#                 "director_token_usage": self.director_token_usage
#             })
#         state.chapter_plan = chapter_plan
#         state.next_action = "scene_planner"
#         return state.__dict__