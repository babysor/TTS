import os

from trainer import Trainer, TrainerArgs
from dataclasses import field

from TTS.config import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.fast_speech_config import FastSpeechConfig
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager



if __name__ == '__main__':
    # output_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.abspath("D:/audiodata/coqui1/")

    # init configs
    dataset_config = BaseDatasetConfig(
        name="aidatatang_200zh",
        meta_file_train="transcript/aidatatang_200_zh_transcript.txt",
        path=os.path.join(output_path, "../aidatatang_200zh/"),
    )

    audio_config = BaseAudioConfig(
        sample_rate=16000,
        do_trim_silence=True,
        trim_db=60.0,
        signal_norm=False,
        mel_fmin=0.0,
        mel_fmax=8000,
        spec_gain=1.0,
        log_func="np.log",
        ref_level_db=20,
        preemphasis=0.0,
    )

    config = FastSpeechConfig(
        run_name="fast_speech_aidatatang",
        audio=audio_config,
        batch_size=28,
        eval_batch_size=16,
        num_loader_workers=8,
        num_eval_loader_workers=4,
        compute_input_seq_cache=True,
        use_speaker_embedding=True,
        compute_f0=False,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="chinese_mandarin_cleaners",
        use_phonemes=True,
        phoneme_language="zh-cn",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        precompute_num_workers=8,
        print_step=50,
        print_eval=False,
        mixed_precision=False,
        max_seq_len=500000,
        output_path=output_path,
        datasets=[dataset_config],
        optimizer_params={"betas": [0.9, 0.99], "weight_decay": 0},
        lr_scheduler="MultiStepLR", 
        lr=1e-5,   
        lr_scheduler_params={
            "gamma": 0.5,
            "milestones": [250000, 320000, 400000]
        },
        test_sentences=[
            "测试一下声音，中文试试.",
        ],
        characters=CharactersConfig(
            pad="_",
            eos="~",
            bos="^",
            blank=None,
            characters="",
            # will be replaced by phonemes if use_phonemes is true
            punctuations="\uff0c\u3002\uff1f\uff01\uff5e\uff1a\uff1b*\u2014\u2014-\uff08\uff09\u3010\u3011!'(),-.:;? ",
            phonemes="ABCDEFGHIJKLMNOPQRSTUVWXYZ12345giy\u0268\u0289\u026fu\u026a\u028f\u028ae\u00f8\u0258\u0259\u0275\u0264o\u025b\u0153\u025c\u025e\u028c\u0254\u00e6\u0250a\u0276\u0251\u0252\u1d7b\u0298\u0253\u01c0\u0257\u01c3\u0284\u01c2\u0260\u01c1\u029bpbtd\u0288\u0256c\u025fk\u0261q\u0262\u0294\u0274\u014b\u0272\u0273n\u0271m\u0299r\u0280\u2c71\u027e\u027d\u0278\u03b2fv\u03b8\u00f0sz\u0283\u0292\u0282\u0290\u00e7\u029dx\u0263\u03c7\u0281\u0127\u0295h\u0266\u026c\u026e\u028b\u0279\u027bj\u0270l\u026d\u028e\u029f\u02c8\u02cc\u02d0\u02d1\u028dw\u0265\u029c\u02a2\u02a1\u0255\u0291\u027a\u0267\u025a\u02de\u026b",
            is_unique=False,
            is_sorted=True
        ),
    )


    ## INITIALIZE THE AUDIO PROCESSOR
    # Audio processor is used for feature extraction and audio I/O.
    # It mainly serves to the dataloader and the training loggers.
    ap = AudioProcessor.init_from_config(config)

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # If characters are not defined in the config, default characters are passed to the config
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # LOAD DATA SAMPLES
    # Each sample is a list of ```[text, audio_file_path, speaker_name]```
    # You can define your custom sample loader returning the list of samples.
    # Or define your custom formatter and pass it to the `load_tts_samples`.
    # Check `TTS.tts.datasets.load_tts_samples` for more details.
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init speaker manager for multi-speaker training
    # it maps speaker-id to speaker-name in the model and data-loader
    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
    config.model_args.num_speakers = speaker_manager.num_speakers

    # init model
    model = ForwardTTS(config, ap, tokenizer, speaker_manager=speaker_manager)

    # INITIALIZE THE TRAINER
    # Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
    # distributed training, etc.
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    # AND... 3,2,1... 🚀
    trainer.fit()
