import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms
import tritonclient.http as httpclient


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Send a batch of MNIST images to a Triton inference server."
	)
	parser.add_argument("--url", default="localhost:8000", help="Triton HTTP URL")
	parser.add_argument("--model-name", default="cnn")
	parser.add_argument("--data-dir", default="data/MNIST")
	parser.add_argument("--batch-size", type=int, default=10)
	return parser.parse_args()


def load_batch(data_dir: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
	if batch_size <= 0:
		raise ValueError("batch-size must be greater than zero")

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,)),
	])
	dataset = datasets.MNIST(
		root=Path(data_dir),
		train=False,
		download=True,
		transform=transform,
	)
	images, labels = zip(*(dataset[index] for index in range(batch_size)))
	image_batch = torch.stack(images).numpy().astype(np.float32)
	label_batch = np.asarray(labels, dtype=np.int64)
	return image_batch, label_batch


def infer_batch(
	client: httpclient.InferenceServerClient,
	model_name: str,
	image_batch: np.ndarray,
) -> np.ndarray:
	infer_input = httpclient.InferInput("input", image_batch.shape, "FP32")
	infer_input.set_data_from_numpy(image_batch)
	infer_output = httpclient.InferRequestedOutput("output")

	response = client.infer(
		model_name=model_name,
		inputs=[infer_input],
		outputs=[infer_output],
	)
	output = response.as_numpy("output")
	if output is None or output.shape != (image_batch.shape[0], 10):
		raise RuntimeError(
			f"Unexpected output shape: {None if output is None else output.shape}"
		)
	return output


def main() -> None:
	args = parse_args()
	image_batch, labels = load_batch(args.data_dir, args.batch_size)

	client = httpclient.InferenceServerClient(url=args.url)
	if not client.is_server_ready():
		raise RuntimeError(f"Triton server is not ready at http://{args.url}")
	if not client.is_model_ready(args.model_name):
		raise RuntimeError(f"Model '{args.model_name}' is not ready")

	output = infer_batch(client, args.model_name, image_batch)
	predictions = np.argmax(output, axis=1)
	confidence = np.max(torch.softmax(torch.from_numpy(output), dim=1).numpy(), axis=1)
	accuracy = np.mean(predictions == labels)

	print(f"Sent batch: {image_batch.shape}")
	print(f"Received output: {output.shape}")
	print(f"Predictions: {predictions.tolist()}")
	print(f"Labels:      {labels.tolist()}")
	print(f"Confidence:  {[round(value, 4) for value in confidence.tolist()]}")
	print(f"Batch accuracy: {accuracy:.1%}")


if __name__ == "__main__":
	main()
