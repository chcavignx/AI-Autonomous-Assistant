# openai_model_names  = (
#     # "openai/whisper-large-v2",
#     "openai/whisper-small",
#     "openai/whisper-small.en",
#     # "openai/whisper-medium",
#     # "openai/whisper-medium.en",
#     "openai/whisper-tiny",
#     "openai/whisper-tiny.en",
#     "openai/whisper-base",
#     "openai/whisper-base.en",
#     # "openai/whisper-large",
# )
# for model_name in openai_model_names:
#     print(f"Downloading and saving {model_name} to {cache_dir}")
#     # Download and save Model from Hugging Face Hub
#     snapshot_download(repo_id=model_name, repo_type=repo_type, cache_dir=cache_dir)
#     print(f"Model saved to: {os.path.join(cache_dir, model_name)}")
#     try:
#         cmd = [
#             "ct2-transformers-converter",
#             "--model", model_name,
#             "--output_dir", os.path.join(cache_dir, model_name.replace("/", "--") + "-ct2"),
#             "--copy_files", "tokenizer.json preprocessor_config.json",
#             "--quantization", "float16"
#         ]
#         process = subprocess.Popen(
#             cmd,
#             stdin=subprocess.PIPE,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )
#         stdout, stderr = process.communicate()
#         if process.returncode != 0:
#             print(f"❌ ct2-transformers-converter CLI error: {stderr.decode()}")
#             success = False
#         else:
#             success = True
#     except Exception as e:
#             print(f"❌ ct2-transformers-converter CLI error: {e}")
#             success = False
#     if success:
#         print(f"✅ ct2-transformers-converter CLI succeeded for {model_name}")
#         print(f"Model saved to: {os.path.join(cache_dir, model_name.replace('/', '--') + '-ct2')}")
#     else:
#         print(f"❌ ct2-transformers-converter CLI failed for {model_name}")
# Requirements:
# pip install huggingface-hub
# pip install faster-whisper
# pip install transformers[torch]>=4.23

# ct2-transformers-converter --model openai/whisper-large-v3 --output_dir whisper-large-v3-ct2
# --copy_files tokenizer.json preprocessor_config.json --quantization float16
