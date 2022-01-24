import csv

def test(record_base_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    model.load_state_dict(torch.load('/content/gdrive/MyDrive/checkpoint_run_1_wavelet_mexh_lead0.pt'))
    model.eval()
    test_set_path = '/content/gdrive/MyDrive/test_data'
    test_ref_path = '/content/gdrive/MyDrive/REFERENCE.csv'
    test_data = ArrhythmiaDataset(test_set_path, test_ref_path, leads=0, normalize=True,
                                  smoothen=True, wavelet='mexh', testing=True)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=False, num_workers=1)

    with open('answers.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Recording', 'Result'])
        for batch in test_loader:
            images = batch[0].to(device)
            names = batch[1]
            preds = model(images)
            result = torch.argmax(preds, dim=1) + 1
            # result = random.randint(1, 9)
            ## If the classification result is an invalid number, the result will be determined as normal(1).
            for i in range(len(names)):
              answer = [names[i], result[i].item()]
              if answer[1] > 9 or answer[1] < 1 or np.isnan(answer[1]):
                  answer[1] = 1

              writer.writerow(answer)
    csvfile.close()

if __name__ == '__main__':
    result = test(record_base_path='/content/gdrive/MyDrive/test_data/')
