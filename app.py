from src.pipelines.train_pipeline import train_pipeline_definition
from src.utils.common_utils import load_config, parse_app_args
from src.utils.model_utils import load_params

if __name__ == "__main__":
    args = parse_app_args()
    config = load_config(args.config)
    params = load_params(args.params)
    if args.selected_segments:
        selected_segments = args.selected_segments
    else:
        selected_segments = config['data']['segments']
    print(f"🚀 Starting training pipeline for segments: {selected_segments}")
    train_pipeline_definition(config, params, selected_segments)
