#!/bin/bash

echo Downloading dataset to $1 ...
wget -O $1/ego4d_for_tuning_forecasting.json "https://www.dropbox.com/scl/fi/bmv4wyxcyjr5pwhpjtg4s/ego4d_for_tuning_forecasting.json?rlkey=ytm49eohahfxkagl1yvbrde2e&st=y5xeds06&dl=1"
wget -O $1/epickitchen_for_tuning_forecasting.json "https://www.dropbox.com/scl/fi/5h0wv57wukbypdjmus72d/epickitchen_for_tuning_forecasting.json?rlkey=vrxz9ww6tkefbzxsjflxfnsha&st=hfi9hssh&dl=1"
