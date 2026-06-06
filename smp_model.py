import torch
import segmentation_models_pytorch as smp

def main():
    # Konfiguracja urządzenia pod Apple Silicon (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Wybrano urządzenie: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Wybrano urządzenie: CUDA")
    else:
        device = torch.device("cpu")
        print("Wybrano urządzenie: CPU")

    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=3,                      
    )

    model = model.to(device)
    print(f"\nModel pomyślnie zainicjowany i przeniesiony na {device}!")


    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    

    output = model(dummy_input)

    print("\n--- TEST MODELU ---")
    print(f"Kształt danych wejściowych: {dummy_input.shape}")
    print(f"Kształt danych wyjściowych: {output.shape}") 
    
if __name__ == '__main__':
    main()
