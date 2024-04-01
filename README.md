
# X-Raydar Official Repository

[![x-raydar](https://www.x-raydar.info/img/logos/logo-online.png)](https://www.x-raydar.info/)

## Development of a freely accessible deep learning platform for comprehensive chest X-ray reading: a retrospective multicenter study in the UK

The code in this repository refers to the paper published on "The Lancet Digital Health" journal.

### Testing the model

#### NOTE: This is not for clinical use ####

1. Clone this repository
2. Register on [x-raydar](https://www.x-raydar.info/) official webpage and accept our terms and conditions
3. Download the network weights and add them in \src\model_20210820_XNet38MS\model_weights
4. Use the DICOM in \demo_data to test the model



In order to download the pretrained network weights you will need to first register on 
```http
  https://www.x-raydar.info/
``` 
and accept our terms and conditions. 

## Code Example

``` python
model = predict.build_model()

filename = '../demo_data/04f72062c19d9cd7a55519708aa2cc58b5e52b52' # test DICOM

dicom = pydicom.read_file(filename)
image_original = dicom_utils.img_clean(dicom)
predict.main(image_original, model)
```


# Contact

For questions, suggestions, or collaborations, please contact Giovanni Montana at g.montana@warwick.ac.uk.

