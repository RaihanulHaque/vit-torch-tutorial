def check_tf_gpu():
    """Checks for available GPUs and prints information about them."""
    import tensorflow as tf

    try:
        # Check if TensorFlow was built with GPU support
        if tf.config.list_physical_devices('GPU'):
            print("TensorFlow was built with GPU support.")

            # Get a list of physical GPUs
            gpus = tf.config.list_physical_devices('GPU')

            if gpus:
                print(f"Found {len(gpus)} GPU(s):")
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu}")

                    # Get details about the specific GPU (requires some more setup, see below)
                    try:
                        gpu_details = tf.config.experimental.get_device_details(gpu)
                        print(f"    Details: {gpu_details}") # This might not always give much info.
                    except Exception as e:
                        print(f"    Could not retrieve detailed info (May require CUDA setup): {e}")


            else:
                print("No GPUs found by TensorFlow, even though it was built with GPU support. Check your CUDA/cuDNN installation.")  # Important!

        else:
            print("TensorFlow was NOT built with GPU support.")
            print("If you have a GPU, you may need to install the correct TensorFlow version (e.g., tensorflow-gpu).")

    except Exception as e:
        print(f"An error occurred: {e}")


def check_torch_gpu():
    """Checks for available GPUs using PyTorch and prints information about them."""
    import torch

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('CUDA:',torch.version.cuda)

        cudnn = torch.backends.cudnn.version()
        cudnn_major = cudnn // 1000
        cudnn = cudnn % 1000
        cudnn_minor = cudnn // 100
        cudnn_patch = cudnn % 100
        print( 'cuDNN:', '.'.join([str(cudnn_major),str(cudnn_minor),str(cudnn_patch)]) )
        


if __name__ == "__main__":
    check_tf_gpu()
    # check_torch_gpu()