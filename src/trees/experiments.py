from src.models import (
    AttemptEdge,
    FixAttemptConfig,
    FixPromptConfig,
    KTopConfig,
    LLMConfig,
    Model,
    PoolingConfig,
    Prompt,
    RootAttemptConfig,
    RootPromptConfig,
)

# TODO: Change model here
# model = Model.llama_3_1_8b_instruct
# model = Model.qwen2_5_14b
# model = Model.qwen2_5_code_32b
# model = Model.qwen2_5_72b
# model = Model.code_lamma
# model = Model.claude_3_5_sonnet
# model = Model.gpt_4o
# model = Model.claude_3_5_sonnet
# model = Model.gpt_4o_mini
# model = Model.gemini_1_5_pro
# model = Model.gemini_1_5_pro
model = Model.qwen2_5_72b_int4


# fast_test: list[RootAttemptConfig] = [
#     RootAttemptConfig(
#         include_all_attempts_in_fixes=True,
#         attempts=3,  # Reduced to just 3 attempts
#         llm_config=LLMConfig(
#             model=model,
#             temperature=0.95,
#         ),
#         prompt_config=RootPromptConfig(
#             base_prompt=Prompt.REASONING,
#             use_examples=True,
#             use_diffs=True,
#             use_images=False,
#             use_ascii=True,
#             use_array=True,
#         ),
#         fixes=[],  # No fix level - just one level of attempts
#     ),
# ]


# fast_test_with_fix: list[RootAttemptConfig] = [
#     RootAttemptConfig(
#         include_all_attempts_in_fixes=True,
#         attempts=3,  # 3 initial attempts
#         llm_config=LLMConfig(
#             model=model,
#             temperature=0.95,
#         ),
#         prompt_config=RootPromptConfig(
#             base_prompt=Prompt.REASONING,
#             use_examples=True,
#             use_diffs=True,
#             use_images=True,
#             use_ascii=True,
#             use_array=True,
#             use_image=True,
#         ),
#         fixes=[
#             AttemptEdge(
#                 k_top_config=KTopConfig(
#                     k_top=2,  # Take top 2 attempts from initial level
#                     unique_code=False,
#                     unique_output=False
#                 ),
#                 configs=[
#                     FixAttemptConfig(
#                         attempts=2,  # 2 fix attempts for each selected solution
#                         llm_config=LLMConfig(
#                             model=model,
#                             temperature=0.95,
#                         ),
#                         prompt_config=FixPromptConfig(
#                             base_prompt=Prompt.REASONING,
#                             use_ascii=True,
#                             use_array=True,
#                             use_image=True,
#                             use_fix_reasoning_tags=True,
#                             use_fix_fail_line=True,
#                             use_typical_issue_text=True,
#                             include_diffs=True,
#                         ),
#                         fixes=[],  # No more fixes after this level
#                     )
#                 ],
#             ),
#         ],
#     ),
# ]



small_deep: list[RootAttemptConfig] = [
    RootAttemptConfig(
        include_all_attempts_in_fixes=True,
        attempts=10,
        llm_config=LLMConfig(
            model=model,
            temperature=0.95,
        ),
        prompt_config=RootPromptConfig(
            base_prompt=Prompt.REASONING,
            use_examples=True,
            use_diffs=True,
            use_images=True,
            use_ascii=True,
            use_array=True,
            use_image=True,
        ),
        fixes=[
            AttemptEdge(
                k_top_config=KTopConfig(
                    k_top=5, unique_code=False, unique_output=False
                ),
                configs=[
                    FixAttemptConfig(
                        attempts=2,
                        llm_config=LLMConfig(
                            model=model,
                            temperature=0.95,
                        ),
                        prompt_config=FixPromptConfig(
                            base_prompt=Prompt.REASONING,
                            use_ascii=True,
                            use_array=True,
                            use_image=True,
                            use_fix_reasoning_tags=True,
                            use_fix_fail_line=True,
                            use_typical_issue_text=True,
                            include_diffs=True,
                        ),
                        fixes=[
                            AttemptEdge(
                                k_top_config=KTopConfig(
                                    k_top=5, unique_code=False, unique_output=False
                                ),
                                configs=[
                                    FixAttemptConfig(
                                        attempts=2,
                                        llm_config=LLMConfig(
                                            model=model,
                                            temperature=0.95,
                                        ),
                                        prompt_config=FixPromptConfig(
                                            base_prompt=Prompt.REASONING,
                                            use_ascii=True,
                                            use_array=True,
                                            use_image=True,
                                            use_fix_reasoning_tags=True,
                                            use_fix_fail_line=True,
                                            use_typical_issue_text=True,
                                            include_diffs=True,
                                        ),
                                        fixes=[
                                            AttemptEdge(
                                                k_top_config=KTopConfig(
                                                    k_top=5,
                                                    unique_code=False,
                                                    unique_output=False,
                                                ),
                                                configs=[
                                                    FixAttemptConfig(
                                                        attempts=2,
                                                        llm_config=LLMConfig(
                                                            model=model,
                                                            temperature=0.95,
                                                        ),
                                                        prompt_config=FixPromptConfig(
                                                            base_prompt=Prompt.REASONING,
                                                            use_ascii=True,
                                                            use_array=True,
                                                            use_image=True,
                                                            use_fix_reasoning_tags=True,
                                                            use_fix_fail_line=True,
                                                            use_typical_issue_text=True,
                                                            include_diffs=True,
                                                        ),
                                                        fixes=[
                                                            AttemptEdge(
                                                                k_top_config=KTopConfig(
                                                                    k_top=5,
                                                                    unique_code=False,
                                                                    unique_output=False,
                                                                ),
                                                                configs=[
                                                                    FixAttemptConfig(
                                                                        attempts=2,
                                                                        llm_config=LLMConfig(
                                                                            model=model,
                                                                            temperature=0.95,
                                                                        ),
                                                                        prompt_config=FixPromptConfig(
                                                                            base_prompt=Prompt.REASONING,
                                                                            use_ascii=True,
                                                                            use_array=True,
                                                                            use_image=True,
                                                                            use_fix_reasoning_tags=True,
                                                                            use_fix_fail_line=True,
                                                                            use_typical_issue_text=True,
                                                                            include_diffs=True,
                                                                        ),
                                                                        fixes=[
                                                                            AttemptEdge(
                                                                                k_top_config=KTopConfig(
                                                                                    k_top=5,
                                                                                    unique_code=False,
                                                                                    unique_output=False,
                                                                                ),
                                                                                configs=[
                                                                                    FixAttemptConfig(
                                                                                        attempts=2,
                                                                                        llm_config=LLMConfig(
                                                                                            model=model,
                                                                                            temperature=0.95,
                                                                                        ),
                                                                                        prompt_config=FixPromptConfig(
                                                                                            base_prompt=Prompt.REASONING,
                                                                                            use_ascii=True,
                                                                                            use_array=True,
                                                                                            use_image=True,
                                                                                            use_fix_reasoning_tags=True,
                                                                                            use_fix_fail_line=True,
                                                                                            use_typical_issue_text=True,
                                                                                            include_diffs=True,
                                                                                        ),
                                                                                        fixes=[
                                                                                            AttemptEdge(
                                                                                                k_top_config=KTopConfig(
                                                                                                    k_top=5,
                                                                                                    unique_code=False,
                                                                                                    unique_output=False,
                                                                                                ),
                                                                                                configs=[
                                                                                                    FixAttemptConfig(
                                                                                                        attempts=2,
                                                                                                        llm_config=LLMConfig(
                                                                                                            model=model,
                                                                                                            temperature=0.95,
                                                                                                        ),
                                                                                                        prompt_config=FixPromptConfig(
                                                                                                            base_prompt=Prompt.REASONING,
                                                                                                            use_ascii=True,
                                                                                                            use_array=True,
                                                                                                            use_image=True,
                                                                                                            use_fix_reasoning_tags=True,
                                                                                                            use_fix_fail_line=True,
                                                                                                            use_typical_issue_text=True,
                                                                                                            include_diffs=True,
                                                                                                        ),
                                                                                                        fixes=[],
                                                                                                    )
                                                                                                ],
                                                                                            )
                                                                                        ],
                                                                                    )
                                                                                ],
                                                                            )
                                                                        ],
                                                                    )
                                                                ],
                                                            )
                                                        ],
                                                    )
                                                ],
                                            )
                                        ],
                                    )
                                ],
                            )
                        ],
                    )
                ],
            ),
        ],
    ),
]

deep: list[RootAttemptConfig] = [
    RootAttemptConfig(
        include_all_attempts_in_fixes=True,
        attempts=50,
        llm_config=LLMConfig(
            model=model,
            temperature=0.95,
        ),
        prompt_config=RootPromptConfig(
            base_prompt=Prompt.REASONING,
            use_examples=True,
            use_diffs=True,
            use_images=True,
            use_ascii=True,
            use_array=True,
            use_image=True,
        ),
        fixes=[
            AttemptEdge(
                k_top_config=KTopConfig(
                    k_top=5, unique_code=False, unique_output=False
                ),
                configs=[
                    FixAttemptConfig(
                        attempts=10,
                        llm_config=LLMConfig(
                            model=model,
                            temperature=0.95,
                        ),
                        prompt_config=FixPromptConfig(
                            base_prompt=Prompt.REASONING,
                            use_ascii=True,
                            use_array=True,
                            use_image=True,
                            use_fix_reasoning_tags=True,
                            use_fix_fail_line=True,
                            use_typical_issue_text=True,
                            include_diffs=True,
                        ),
                        fixes=[
                            AttemptEdge(
                                k_top_config=KTopConfig(
                                    k_top=5, unique_code=False, unique_output=False
                                ),
                                configs=[
                                    FixAttemptConfig(
                                        attempts=10,
                                        llm_config=LLMConfig(
                                            model=model,
                                            temperature=0.95,
                                        ),
                                        prompt_config=FixPromptConfig(
                                            base_prompt=Prompt.REASONING,
                                            use_ascii=True,
                                            use_array=True,
                                            use_image=True,
                                            use_fix_reasoning_tags=True,
                                            use_fix_fail_line=True,
                                            use_typical_issue_text=True,
                                            include_diffs=True,
                                        ),
                                        fixes=[
                                            AttemptEdge(
                                                k_top_config=KTopConfig(
                                                    k_top=5,
                                                    unique_code=False,
                                                    unique_output=False,
                                                ),
                                                configs=[
                                                    FixAttemptConfig(
                                                        attempts=10,
                                                        llm_config=LLMConfig(
                                                            model=model,
                                                            temperature=0.95,
                                                        ),
                                                        prompt_config=FixPromptConfig(
                                                            base_prompt=Prompt.REASONING,
                                                            use_ascii=True,
                                                            use_array=True,
                                                            use_image=True,
                                                            use_fix_reasoning_tags=True,
                                                            use_fix_fail_line=True,
                                                            use_typical_issue_text=True,
                                                            include_diffs=True,
                                                        ),
                                                        fixes=[
                                                            AttemptEdge(
                                                                k_top_config=KTopConfig(
                                                                    k_top=5,
                                                                    unique_code=False,
                                                                    unique_output=False,
                                                                ),
                                                                configs=[
                                                                    FixAttemptConfig(
                                                                        attempts=10,
                                                                        llm_config=LLMConfig(
                                                                            model=model,
                                                                            temperature=0.95,
                                                                        ),
                                                                        prompt_config=FixPromptConfig(
                                                                            base_prompt=Prompt.REASONING,
                                                                            use_ascii=True,
                                                                            use_array=True,
                                                                            use_image=True,
                                                                            use_fix_reasoning_tags=True,
                                                                            use_fix_fail_line=True,
                                                                            use_typical_issue_text=True,
                                                                            include_diffs=True,
                                                                        ),
                                                                        fixes=[
                                                                            AttemptEdge(
                                                                                k_top_config=KTopConfig(
                                                                                    k_top=5,
                                                                                    unique_code=False,
                                                                                    unique_output=False,
                                                                                ),
                                                                                configs=[
                                                                                    FixAttemptConfig(
                                                                                        attempts=10,
                                                                                        llm_config=LLMConfig(
                                                                                            model=model,
                                                                                            temperature=0.95,
                                                                                        ),
                                                                                        prompt_config=FixPromptConfig(
                                                                                            base_prompt=Prompt.REASONING,
                                                                                            use_ascii=True,
                                                                                            use_array=True,
                                                                                            use_image=True,
                                                                                            use_fix_reasoning_tags=True,
                                                                                            use_fix_fail_line=True,
                                                                                            use_typical_issue_text=True,
                                                                                            include_diffs=True,
                                                                                        ),
                                                                                        fixes=[
                                                                                            AttemptEdge(
                                                                                                k_top_config=KTopConfig(
                                                                                                    k_top=5,
                                                                                                    unique_code=False,
                                                                                                    unique_output=False,
                                                                                                ),
                                                                                                configs=[
                                                                                                    FixAttemptConfig(
                                                                                                        attempts=10,
                                                                                                        llm_config=LLMConfig(
                                                                                                            model=model,
                                                                                                            temperature=0.95,
                                                                                                        ),
                                                                                                        prompt_config=FixPromptConfig(
                                                                                                            base_prompt=Prompt.REASONING,
                                                                                                            use_ascii=True,
                                                                                                            use_array=True,
                                                                                                            use_image=True,
                                                                                                            use_fix_reasoning_tags=True,
                                                                                                            use_fix_fail_line=True,
                                                                                                            use_typical_issue_text=True,
                                                                                                            include_diffs=True,
                                                                                                        ),
                                                                                                        fixes=[],
                                                                                                    )
                                                                                                ],
                                                                                            )
                                                                                        ],
                                                                                    )
                                                                                ],
                                                                            )
                                                                        ],
                                                                    )
                                                                ],
                                                            )
                                                        ],
                                                    )
                                                ],
                                            )
                                        ],
                                    )
                                ],
                            )
                        ],
                    )
                ],
            ),
        ],
    ),
]
deep_pool: list[RootAttemptConfig] = [
    RootAttemptConfig(
        include_all_attempts_in_fixes=True,
        attempts=50,
        llm_config=LLMConfig(
            model=model,
            temperature=0.95,
        ),
        prompt_config=RootPromptConfig(
            base_prompt=Prompt.REASONING,
            use_examples=True,
            use_diffs=True,
            use_images=True,
            use_ascii=True,
            use_array=True,
            use_image=True,
        ),
        fixes=[
            AttemptEdge(
                pooling=(PoolingConfig(size=5)),
                k_top_config=KTopConfig(
                    k_top=5, unique_code=False, unique_output=False
                ),
                configs=[
                    FixAttemptConfig(
                        attempts=10,
                        llm_config=LLMConfig(
                            model=Model.claude_3_5_sonnet,
                            temperature=0.95,
                        ),
                        prompt_config=FixPromptConfig(
                            base_prompt=Prompt.REASONING,
                            use_ascii=True,
                            use_array=True,
                            use_image=True,
                            use_fix_reasoning_tags=True,
                            use_fix_fail_line=True,
                            use_typical_issue_text=True,
                            include_diffs=True,
                        ),
                        fixes=[
                            AttemptEdge(
                                pooling=(PoolingConfig(size=5)),
                                k_top_config=KTopConfig(
                                    k_top=5, unique_code=False, unique_output=False
                                ),
                                configs=[
                                    FixAttemptConfig(
                                        attempts=10,
                                        llm_config=LLMConfig(
                                            model=Model.claude_3_5_sonnet,
                                            temperature=0.95,
                                        ),
                                        prompt_config=FixPromptConfig(
                                            base_prompt=Prompt.REASONING,
                                            use_ascii=True,
                                            use_array=True,
                                            use_image=True,
                                            use_fix_reasoning_tags=True,
                                            use_fix_fail_line=True,
                                            use_typical_issue_text=True,
                                            include_diffs=True,
                                        ),
                                        fixes=[
                                            AttemptEdge(
                                                pooling=(PoolingConfig(size=5)),
                                                k_top_config=KTopConfig(
                                                    k_top=5,
                                                    unique_code=False,
                                                    unique_output=False,
                                                ),
                                                configs=[
                                                    FixAttemptConfig(
                                                        attempts=10,
                                                        llm_config=LLMConfig(
                                                            model=Model.claude_3_5_sonnet,
                                                            temperature=0.95,
                                                        ),
                                                        prompt_config=FixPromptConfig(
                                                            base_prompt=Prompt.REASONING,
                                                            use_ascii=True,
                                                            use_array=True,
                                                            use_image=True,
                                                            use_fix_reasoning_tags=True,
                                                            use_fix_fail_line=True,
                                                            use_typical_issue_text=True,
                                                            include_diffs=True,
                                                        ),
                                                        fixes=[
                                                            AttemptEdge(
                                                                pooling=(
                                                                    PoolingConfig(
                                                                        size=3
                                                                    )
                                                                ),
                                                                k_top_config=KTopConfig(
                                                                    k_top=5,
                                                                    unique_code=False,
                                                                    unique_output=False,
                                                                ),
                                                                configs=[
                                                                    FixAttemptConfig(
                                                                        attempts=10,
                                                                        llm_config=LLMConfig(
                                                                            model=Model.claude_3_5_sonnet,
                                                                            temperature=0.95,
                                                                        ),
                                                                        prompt_config=FixPromptConfig(
                                                                            base_prompt=Prompt.REASONING,
                                                                            use_ascii=True,
                                                                            use_array=True,
                                                                            use_image=True,
                                                                            use_fix_reasoning_tags=True,
                                                                            use_fix_fail_line=True,
                                                                            use_typical_issue_text=True,
                                                                            include_diffs=True,
                                                                        ),
                                                                        fixes=[
                                                                            AttemptEdge(
                                                                                pooling=(
                                                                                    PoolingConfig(
                                                                                        size=3
                                                                                    )
                                                                                ),
                                                                                k_top_config=KTopConfig(
                                                                                    k_top=5,
                                                                                    unique_code=False,
                                                                                    unique_output=False,
                                                                                ),
                                                                                configs=[
                                                                                    FixAttemptConfig(
                                                                                        attempts=10,
                                                                                        llm_config=LLMConfig(
                                                                                            model=Model.claude_3_5_sonnet,
                                                                                            temperature=0.95,
                                                                                        ),
                                                                                        prompt_config=FixPromptConfig(
                                                                                            base_prompt=Prompt.REASONING,
                                                                                            use_ascii=True,
                                                                                            use_array=True,
                                                                                            use_image=True,
                                                                                            use_fix_reasoning_tags=True,
                                                                                            use_fix_fail_line=True,
                                                                                            use_typical_issue_text=True,
                                                                                            include_diffs=True,
                                                                                        ),
                                                                                        fixes=[
                                                                                            AttemptEdge(
                                                                                                pooling=(
                                                                                                    PoolingConfig(
                                                                                                        size=3
                                                                                                    )
                                                                                                ),
                                                                                                k_top_config=KTopConfig(
                                                                                                    k_top=5,
                                                                                                    unique_code=False,
                                                                                                    unique_output=False,
                                                                                                ),
                                                                                                configs=[
                                                                                                    FixAttemptConfig(
                                                                                                        attempts=10,
                                                                                                        llm_config=LLMConfig(
                                                                                                            model=Model.claude_3_5_sonnet,
                                                                                                            temperature=0.95,
                                                                                                        ),
                                                                                                        prompt_config=FixPromptConfig(
                                                                                                            base_prompt=Prompt.REASONING,
                                                                                                            use_ascii=True,
                                                                                                            use_array=True,
                                                                                                            use_image=True,
                                                                                                            use_fix_reasoning_tags=True,
                                                                                                            use_fix_fail_line=True,
                                                                                                            use_typical_issue_text=True,
                                                                                                            include_diffs=True,
                                                                                                        ),
                                                                                                        fixes=[],
                                                                                                    )
                                                                                                ],
                                                                                            )
                                                                                        ],
                                                                                    )
                                                                                ],
                                                                            )
                                                                        ],
                                                                    )
                                                                ],
                                                            )
                                                        ],
                                                    )
                                                ],
                                            )
                                        ],
                                    )
                                ],
                            )
                        ],
                    )
                ],
            ),
        ],
    ),
]
grid_only: list[RootAttemptConfig] = [
    RootAttemptConfig(
        include_all_attempts_in_fixes=True,
        attempts=20,
        llm_config=LLMConfig(
            model=Model.gemini_1_5_pro,
            temperature=0.95,
        ),
        prompt_config=RootPromptConfig(
            base_prompt=Prompt.ONLY_GRID,
            use_examples=False,
            use_diffs=False,
            use_images=False,
            use_ascii=False,
            use_array=True,
            use_image=False,
        ),
        fixes=[],
    ),
]


sonnet_writeup_shallow: list[RootAttemptConfig] = [
    RootAttemptConfig(
        include_all_attempts_in_fixes=True,
        attempts=50,
        llm_config=LLMConfig(
            model=model,
            temperature=0.95,
        ),
        prompt_config=RootPromptConfig(
            base_prompt=Prompt.REASONING,
            use_examples=True,
            use_diffs=True,
            use_images=True,
            use_ascii=True,
            use_array=True,
            use_image=True,
        ),
        fixes=[],
    ),
    RootAttemptConfig(
        include_all_attempts_in_fixes=True,
        attempts=50,
        llm_config=LLMConfig(
            model=model,
            temperature=0.95,
        ),
        prompt_config=RootPromptConfig(
            base_prompt=Prompt.REASONING,
            use_examples=True,
            use_diffs=True,
            use_images=True,
            use_ascii=True,
            use_array=True,
            use_image=True,
        ),
        fixes=[],
    ),
    RootAttemptConfig(
        include_all_attempts_in_fixes=True,
        attempts=50,
        llm_config=LLMConfig(
            model=model,
            temperature=0.95,
        ),
        prompt_config=RootPromptConfig(
            base_prompt=Prompt.REASONING,
            use_examples=True,
            use_diffs=True,
            use_images=True,
            use_ascii=True,
            use_array=True,
            use_image=True,
        ),
        fixes=[],
    ),
    RootAttemptConfig(
        include_all_attempts_in_fixes=True,
        attempts=50,
        llm_config=LLMConfig(
            model=model,
            temperature=0.95,
        ),
        prompt_config=RootPromptConfig(
            base_prompt=Prompt.REASONING,
            use_examples=True,
            use_diffs=True,
            use_images=True,
            use_ascii=True,
            use_array=True,
            use_image=True,
        ),
        fixes=[],
    ),
]
sonnet_writeup_deep: list[RootAttemptConfig] = [
    RootAttemptConfig(
        include_all_attempts_in_fixes=True,
        attempts=50,
        llm_config=LLMConfig(
            model=model,
            temperature=0.95,
        ),
        prompt_config=RootPromptConfig(
            base_prompt=Prompt.REASONING,
            use_examples=True,
            use_diffs=True,
            use_images=True,
            use_ascii=True,
            use_array=True,
            use_image=True,
        ),
        fixes=[
            AttemptEdge(
                k_top_config=KTopConfig(
                    k_top=5, unique_code=False, unique_output=False
                ),
                configs=[
                    FixAttemptConfig(
                        attempts=10,
                        llm_config=LLMConfig(
                            model=model,
                            temperature=0.95,
                        ),
                        prompt_config=FixPromptConfig(
                            base_prompt=Prompt.REASONING,
                            use_ascii=True,
                            use_array=True,
                            use_image=True,
                            use_fix_reasoning_tags=True,
                            use_fix_fail_line=True,
                            use_typical_issue_text=True,
                            include_diffs=True,
                        ),
                        fixes=[
                            AttemptEdge(
                                k_top_config=KTopConfig(
                                    k_top=5, unique_code=False, unique_output=False
                                ),
                                configs=[
                                    FixAttemptConfig(
                                        attempts=10,
                                        llm_config=LLMConfig(
                                            model=model,
                                            temperature=0.95,
                                        ),
                                        prompt_config=FixPromptConfig(
                                            base_prompt=Prompt.REASONING,
                                            use_ascii=True,
                                            use_array=True,
                                            use_image=True,
                                            use_fix_reasoning_tags=True,
                                            use_fix_fail_line=True,
                                            use_typical_issue_text=True,
                                            include_diffs=True,
                                        ),
                                        fixes=[
                                            AttemptEdge(
                                                k_top_config=KTopConfig(
                                                    k_top=5,
                                                    unique_code=False,
                                                    unique_output=False,
                                                ),
                                                configs=[
                                                    FixAttemptConfig(
                                                        attempts=10,
                                                        llm_config=LLMConfig(
                                                            model=model,
                                                            temperature=0.95,
                                                        ),
                                                        prompt_config=FixPromptConfig(
                                                            base_prompt=Prompt.REASONING,
                                                            use_ascii=True,
                                                            use_array=True,
                                                            use_image=True,
                                                            use_fix_reasoning_tags=True,
                                                            use_fix_fail_line=True,
                                                            use_typical_issue_text=True,
                                                            include_diffs=True,
                                                        ),
                                                        fixes=[],
                                                    )
                                                ],
                                            )
                                        ],
                                    )
                                ],
                            )
                        ],
                    )
                ],
            ),
        ],
    ),
]
sonnet_writeup_med: list[RootAttemptConfig] = [
    RootAttemptConfig(
        include_all_attempts_in_fixes=True,
        attempts=70,
        llm_config=LLMConfig(
            model=model,
            temperature=0.95,
        ),
        prompt_config=RootPromptConfig(
            base_prompt=Prompt.REASONING,
            use_examples=True,
            use_diffs=True,
            use_images=True,
            use_ascii=True,
            use_array=True,
            use_image=True,
        ),
        fixes=[
            AttemptEdge(
                k_top_config=KTopConfig(
                    k_top=5, unique_code=False, unique_output=False
                ),
                configs=[
                    FixAttemptConfig(
                        attempts=13,
                        llm_config=LLMConfig(
                            model=model,
                            temperature=0.95,
                        ),
                        prompt_config=FixPromptConfig(
                            base_prompt=Prompt.REASONING,
                            use_ascii=True,
                            use_array=True,
                            use_image=True,
                            use_fix_reasoning_tags=True,
                            use_fix_fail_line=True,
                            use_typical_issue_text=True,
                            include_diffs=True,
                        ),
                        fixes=[
                            AttemptEdge(
                                k_top_config=KTopConfig(
                                    k_top=5, unique_code=False, unique_output=False
                                ),
                                configs=[
                                    FixAttemptConfig(
                                        attempts=13,
                                        llm_config=LLMConfig(
                                            model=model,
                                            temperature=0.95,
                                        ),
                                        prompt_config=FixPromptConfig(
                                            base_prompt=Prompt.REASONING,
                                            use_ascii=True,
                                            use_array=True,
                                            use_image=True,
                                            use_fix_reasoning_tags=True,
                                            use_fix_fail_line=True,
                                            use_typical_issue_text=True,
                                            include_diffs=True,
                                        ),
                                        fixes=[],
                                    )
                                ],
                            )
                        ],
                    )
                ],
            ),
        ],
    ),
]
sonnet_pooling_example: list[RootAttemptConfig] = [
    RootAttemptConfig(
        include_all_attempts_in_fixes=True,
        attempts=5,
        llm_config=LLMConfig(
            model=model,
            temperature=0.95,
        ),
        prompt_config=RootPromptConfig(
            base_prompt=Prompt.REASONING,
            use_examples=True,
            use_diffs=True,
            use_images=True,
            use_ascii=True,
            use_array=True,
            use_image=True,
        ),
        fixes=[
            AttemptEdge(
                pooling=(PoolingConfig(size=2)),
                k_top_config=KTopConfig(
                    k_top=1, unique_code=False, unique_output=False
                ),
                configs=[
                    FixAttemptConfig(
                        attempts=1,
                        llm_config=LLMConfig(
                            model=model,
                            temperature=0.95,
                        ),
                        prompt_config=FixPromptConfig(
                            base_prompt=Prompt.REASONING,
                            use_ascii=True,
                            use_array=True,
                            use_image=True,
                            use_fix_reasoning_tags=True,
                            use_fix_fail_line=True,
                            use_typical_issue_text=True,
                            include_diffs=True,
                        ),
                        fixes=[],
                    )
                ],
            ),
        ],
    ),
]
