from src.probe_engine import (
    DomainProbeEngine,
    load_domains_from_folder,
    print_similarity_matrix,
    print_cohesion,
    print_ownership,
    print_confusion
)

MODEL_NAME = "distilbert-base-uncased"   # ganti sesuai model offline kamu
DOMAINS_DIR = "domains/senior_high"      # kamu sudah buat ini

engine = DomainProbeEngine(MODEL_NAME, local_files_only=True)

domains = load_domains_from_folder(DOMAINS_DIR)

# Auto tune layer (penting buat GPT-Neo / TinyStories)
best_ratio, best_score = engine.auto_tune_layer(domains)

# Run final probe pakai best_ratio
results = engine.run_full_probe(domains, layer_ratio=best_ratio, do_confusion=True)

print_similarity_matrix(results["similarity_matrix"])
print_cohesion(results["cohesion"])
print_ownership(results["ownership"])
print_confusion(results["confusion"])

print("\nGLOBAL WIN RATE:", results["global_win_rate"] * 100, "%")