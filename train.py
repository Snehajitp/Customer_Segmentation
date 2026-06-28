# import os
# from dotenv import load_dotenv

# loaded = load_dotenv()

# print("load_dotenv() returned:", loaded)
# print("Current working directory:", os.getcwd())
# print("MONGO_DB_URL:", os.getenv("MONGO_DB_URL"))

# from src.pipeline.train_pipeline import TrainPipeline

# pipeline = TrainPipeline()
# pipeline.run_pipeline()

from src.pipeline.train_pipeline import TrainPipeline

if __name__ == "__main__":
    try:
        print("========== STARTING TRAINING ==========")

        pipeline = TrainPipeline()
        pipeline.run_pipeline()

        print("\n===================================")
        print("Training completed successfully!")
        print("===================================")

    except Exception as e:
        print("\n========== TRAINING FAILED ==========")
        print(e)
        raise