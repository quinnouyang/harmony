#!/usr/bin/env python




import argparse
import copy
import itertools
import random
import re
from fractions import Fraction
import logging

from music21.chord import Chord
from music21.clef import BassClef, TrebleClef
from music21.instrument import Piano
from music21.key import Key
from music21.meter import TimeSignature
from music21.note import Note
from music21.pitch import Pitch
from music21.roman import RomanNumeral
from music21.stream import Part, Score, Voice, Measure
from music21.tempo import MetronomeMark

SOPRANO_RANGE = (Pitch("C4"), Pitch("G5"))
ALTO_RANGE = (Pitch("G3"), Pitch("C5"))
TENOR_RANGE = (Pitch("C3"), Pitch("G4"))
BASS_RANGE = (Pitch("E2"), Pitch("C4"))

_MODES = ("major", "natural minor", "harmonic minor",)

_TONICS = {
    "major" : ("I", "vi", "iii", "Imaj7", "vi-7", "iii-7",),
    "natural minor" : ("i", "VI", "III", "i-7", "VImaj7", "IIImaj7",),
    "harmonic minor" : ("i", "VI", "III", "imM7", "VImaj7"),
}
_PREDOM = {
    "major" : ("ii", "IV", "ii-7", "IVmaj7",),
    "natural minor" : ("iio", "iv", "ii/o7", "iv-7",),
    "harmonic minor" : ("iio", "iv", "ii/o7", "iv-7",),
}
_DOM = {
    "major" : ("V", "viio", "V7", "vii/o7",),
    "natural minor" : ("v", "VII", "v-7", "VII7",),
    "harmonic minor" : ("V", "viio", "V7", "viio7",),
}


_INVS_7 = ("7", "65", "43", "42",)
_INVS = ("", "6", "64")



_KEYS = tuple("c g d a e b f# c# f b- e- a- d- g- c-".split())

_RE = re.compile(r"([iv]+)", re.I)

_logger = logging.getLogger(__name__)








def voiceNote(noteName, pitchRange):
    """Generates voicings for a note in a given pitch range.

    Returns a list of `Pitch` objects with the same name as the note that also
    fall within the voice's range.
    """
    lowerOctave = pitchRange[0].octave
    upperOctave = pitchRange[1].octave
    for octave in range(lowerOctave, upperOctave + 1):
        n = Pitch(noteName + str(octave))
        if pitchRange[0] <= n <= pitchRange[1]:
            yield n


def _voiceTriadUnordered(noteNames):
    assert len(noteNames) == 3
    for tenor, alto, soprano in itertools.permutations(noteNames, 3):
        for sopranoNote in voiceNote(soprano, SOPRANO_RANGE):
            altoMin = max((ALTO_RANGE[0], sopranoNote.transpose("-P8")))
            altoMax = min((ALTO_RANGE[1], sopranoNote))
            for altoNote in voiceNote(alto, (altoMin, altoMax)):
                tenorMin = max((TENOR_RANGE[0], altoNote.transpose("-P8")))
                tenorMax = min((TENOR_RANGE[1], altoNote))
                for tenorNote in voiceNote(tenor, (tenorMin, tenorMax)):
                    yield Chord([tenorNote, altoNote, sopranoNote])


def _voiceChord(noteNames):
    assert len(noteNames) == 4
    bass = noteNames.pop(0)
    for chord in _voiceTriadUnordered(noteNames):
        for bassNote in voiceNote(bass, BASS_RANGE):
            if bassNote <= chord.bass():
                chord4 = copy.deepcopy(chord)
                chord4.add(bassNote)
                yield chord4


def voiceChord(key, chord):
    """Generates four-part voicings for a fifth or seventh chord.

    The bass note is kept intact, though other notes (and doublings) are
    allowed to vary between different voicings. Intervals between adjacent
    non-bass parts are limited to a single octave.
    """
    leadingTone = key.getLeadingTone().name
    noteNames = [pitch.name for pitch in chord.pitches]
    if chord.containsSeventh():
        yield from _voiceChord(noteNames)
    elif chord.inversion() == 2:
        # must double the fifth
        yield from _voiceChord(noteNames + [chord.fifth.name])
    else:
        # double the root
        if chord.root().name != leadingTone:
            yield from _voiceChord(noteNames + [chord.root().name])
        # double the third
        if chord.third.name != leadingTone:
            yield from _voiceChord(noteNames + [chord.third.name])
        # double the fifth
        if chord.fifth.name != leadingTone:
            yield from _voiceChord(noteNames + [chord.fifth.name])
        # option to omit the fifth
        if chord.romanNumeral == "I" and chord.inversion() == 0:
            yield from _voiceChord([chord.root().name] * 3 + [chord.third.name])


def progressionCost(key, chord1, chord2):
    """Computes elements of cost between two chords: contrary motion, etc."""
    cost = 0

    # Overlapping voices
    if (
        chord2[0] > chord1[1]
        or chord2[1] < chord1[0]
        or chord2[1] > chord1[2]
        or chord2[2] < chord1[1]
        or chord2[2] > chord1[3]
        or chord2[3] < chord1[2]
    ):
        cost += 40

    # Avoid big jumps
    diff = [abs(chord1.pitches[i].midi - chord2.pitches[i].midi) for i in range(4)]
    cost += (diff[3] // 3) ** 2 if diff[3] else 1
    cost += diff[2] ** 2 // 3
    cost += diff[1] ** 2 // 3
    cost += diff[0] ** 2 // 50 if diff[0] != 12 else 0

    # Contrary motion is good, parallel fifths are bad
    for i in range(4):
        for j in range(i + 1, 4):
            t1, t2 = chord1.pitches[j], chord2.pitches[j]
            b1, b2 = chord1.pitches[i], chord2.pitches[i]
            if t1 == t2 and b1 == b2:  # No motion
                continue
            i1, i2 = t1.midi - b1.midi, t2.midi - b2.midi
            if i1 % 12 == i2 % 12 == 7:  # Parallel fifth
                cost += 60
            if i1 % 12 == i2 % 12 == 0:  # Parallel octave
                cost += 100
            if i == 0 and j == 3:  # Soprano and bass not contrary
                if (t2 > t1 and b2 > b1) or (t2 < t1 and b2 < b1):
                    cost += 2

    # Chordal 7th should resolve downward or stay
    if chord1.seventh:
        seventhVoice = chord1.pitches.index(chord1.seventh)
        delta = chord2.pitches[seventhVoice].midi - chord1.seventh.midi
        if delta < -2 or delta > 0:
            cost += 100

    # V->I means ti->do or ti->sol
    pitches = key.getPitches()
    pitches[6] = key.getLeadingTone()
    if (
        chord1.root().name
        in (
            pitches[4].name,
            pitches[6].name,
        )
        and chord2.root().name in (pitches[0].name, pitches[5].name)
        and pitches[6].name in chord1.pitchNames
    ):
        voice = chord1.pitchNames.index(pitches[6].name)
        delta = chord2.pitches[voice].midi - chord1.pitches[voice].midi
        if not (delta == 1 or (delta == -4 and voice >= 1 and voice <= 2)):
            cost += 100

    return cost


def chordCost(key, chord):
    """Computes elements of cost that only pertain to a single chord."""
    cost = 0
    if chord.inversion() == 0:
        # Slightly prefer to double the root in a R.P. chord
        if chord.pitchClasses.count(chord.root().pitchClass) <= 1:
            cost += 1
    return cost


def voiceProgression(key, chordProgression):
    """Voices a chord progression in a specified key using DP.

    Follows eighteenth-century voice leading procedures, as guided by the cost
    function defined in the `chordCost` and `progressionCost` functions.
    Returns a list of four-pitch chords, corresponding to successive Roman
    numerals in the chord progression.
    """
    key = Key(key)
    if isinstance(chordProgression, str):
        chordProgression = list(filter(None, chordProgression.split()))

    dp = [{} for _ in chordProgression]
    for i, numeral in enumerate(chordProgression):
        chord = RomanNumeral(numeral, key)
        voicings = voiceChord(key, chord)
        if i == 0:
            for v in voicings:
                dp[0][v.pitches] = (chordCost(key, v), None)
        else:
            for v in voicings:
                best = (float("inf"), None)
                for pv_pitches, (pcost, _) in dp[i - 1].items():
                    pv = Chord(pv_pitches)
                    ccost = pcost + progressionCost(key, pv, v)
                    if ccost < best[0]:
                        best = (ccost, pv_pitches)
                dp[i][v.pitches] = (best[0] + chordCost(key, v), best[1])

    cur, (totalCost, _) = min(dp[-1].items(), key=lambda p: p[1][0])
    ret = []
    for i in reversed(range(len(chordProgression))):
        ret.append(Chord(cur, lyric=chordProgression[i]))
        cur = dp[i][cur][1]
    return list(reversed(ret)), totalCost




def generateScore(key, chords, lengths=None, ts="4/4"):
    """Generates a four-part score from a sequence of chords.

    Soprano and alto parts are displayed on the top (treble) clef, while tenor
    and bass parts are displayed on the bottom (bass) clef, with correct stem
    directions.
    """
    if lengths is None:
        lengths = [1 for _ in chords]
    voices = [Voice([Piano()]) for _ in range(4)]

    upper_ms = []
    lower_ms = []

    for i, (chord, length) in enumerate(zip(chords, lengths), 1):
        bass, tenor, alto, soprano = [
            Note(p, quarterLength=length) for p in chord.pitches
        ]
        bass.addLyric(chord.lyric)
        bass.stemDirection = alto.stemDirection = "down"
        tenor.stemDirection = soprano.stemDirection = "up"
        # voices[0].append(soprano)
        # voices[1].append(alto)
        # voices[2].append(tenor)
        # voices[3].append(bass)

        upper_m = Measure(number = i)
        upper_m.append(Voice([Piano(), soprano]))
        upper_m.append(Voice([Piano(), alto]))
        upper_ms.append(upper_m)

        lower_m = Measure(number = i)
        lower_m.append(Voice([Piano(), tenor]))
        lower_m.append(Voice([Piano(), bass]))
        lower_ms.append(lower_m)

    key = Key(key)
    tempo = MetronomeMark('slow', 15, Note(type='whole'))

    female = Part([key, tempo, TrebleClef(), TimeSignature(ts), *upper_ms])
    male = Part([key, tempo, BassClef(), TimeSignature(ts), *lower_ms])
    score = Score([female, male])
    return score


def generateChorale(chorale, lengths=None, ts="4/4"):
    """Voices a chorale with multiple phrases.

    Each phrase should be placed on a line in the input string, with the key at
    the beginning followed by space-separated roman numerals. For example,

     D: I vi I6 IV I64 V I
     D: I6 V64 I IV6 V I6 V
     D: I IV6 I6 IV I64 V7 vi
     D: I6 V43 I I6 ii65 V I
     A: I IV64 I vi ii6 V7 I
     b: iv6 i64 iv iio6 i64 V7 i
     A: IV IV V I6 ii V65 I
     D: IV6 I V65 I ii65 V7 I
    """
    lines = [line.strip().split(":") for line in chorale.split("\n") if line.strip()]
    progression = []
    for key, chords in lines:
        phrase, _ = voiceProgression(key, chords)
        progression.extend(phrase)
    score = generateScore(key, progression, lengths, ts)
    return score




def numeral(s):
    return _RE.match(s)[1].lower()

def get_numeral(l, n):
    return [c for c in l if numeral(c) == n]


def choose_weighted(choices):

    weights = [1 if c.endswith("7") else 2.5 for c in choices]
    return random.choices(choices, weights)[0]



def generateRandom(mode=None, length=None):

    # import pudb
    # pu.db

    # jason's shit, insert copyright notice stuff lol


    if mode is None:
        mode = random.choice(list(_MODES))

    if not (type(mode) == str and mode in _MODES):
        raise ValueError

    if length is None:
        length = 8

    if not (type(length) == int and length > 0):
        raise ValueError


    chords = [None] * length

    chords[0] = choose_weighted(get_numeral(_TONICS[mode], "i"))
    chords[-2] = choose_weighted(_DOM[mode])
    chords[-1] = choose_weighted(get_numeral(_TONICS[mode], "i"))

    for i in range(1, length - 2):
        if chords[i - 1] in _DOM[mode]:

            if numeral(chords[i - 1]) == "vii":
                chords[i] = choose_weighted(get_numeral(_TONICS[mode], "i"))

            else:
                function_chords = random.choices((_TONICS, _PREDOM, _DOM), [7, 1, 2])[0][mode]

                chords[i] = choose_weighted(function_chords)

        else:

            chords[i] = choose_weighted((*_TONICS[mode], *_PREDOM[mode], *_DOM[mode]))

        if chords[i] == "V7":
            chords[i] = chords[i].removesuffix("7") + random.choice(_INVS_7)
        elif not chords[i].endswith("7"):
            chords[i] = chords[i] + random.choice(_INVS)

    key = random.choice(_KEYS)
    if mode == "major":
        key = key.upper()

    chord_progression = ' '.join(chords)

    durations = ("1 " * length).removesuffix(" ")
    time_signature = "4/4"

    _logger.debug(f"{key=}")
    _logger.debug(f"{chord_progression=}")

    return {
            "key" : key,
            "chord_progression" : chord_progression,
            "durations" : durations,
            "time_signature" : time_signature,
           }










def main():
    parser = argparse.ArgumentParser(
        description="Generates four-part harmony with idiomatic "
        "voice-leading procedures and dynamic programming."
    )
    parser.add_argument(
        "key",
        type=str,
        nargs="?",
        help="the key of the chord progression",
    )
    parser.add_argument(
        "chord_progression",
        type=str,
        nargs="?",
        help='a sequence of roman numeral annotations, e.g., "I I6 IV V43/ii ii V V7 I"',
    )
    parser.add_argument(
        "durations",
        type=str,
        nargs="?",
        help="the associated durations of the chords (in quarter notes)",
    )
    parser.add_argument(
        "time_signature", type=str, nargs="?", help="the time signature"
    )

    parser.add_argument(
        "-d", "--debug", action="store_true",
    )

    group = parser.add_argument_group("configuration for randomizer")

    group.add_argument(
            "-l", "--length", type=int, help="how many chords",
    )
    group.add_argument(
            "-m", "--mode", choices=_MODES, help="which mode",
    )

    """
    parser.set_defaults(
        key="B-",
        chord_progression="I I6 IV V43/ii ii V V7 I",
        durations="1 1/2 1 1/2 1 1/2 1/2 1",
        time_signature="6/8",
    )
    """


    args = vars(parser.parse_args())

    logging.basicConfig(level=(logging.DEBUG if args.pop("debug") else logging.WARNING))

    mode = args.pop("mode", None)
    length = args.pop("length", None)

    if not any(args.values()):
        args = generateRandom(mode, length)

    key_and_chords = f"{args['key']}: {args['chord_progression']}"
    durations = [Fraction(x) for x in args["durations"].split()]
    time_signature = args["time_signature"]




    chorale = generateChorale(key_and_chords, durations, time_signature)


    visual_method = "musicxml"
    audio_method = "midi"
    visual_method = "text"
    audio_method = "text"


    chorale.measure(2).show(visual_method)
    chorale.show(audio_method) # use a hidden audio player or something

 
    while True:
        answer = input("Ready for answer [y/n]: ").lower()

        if answer in {"y", "n"}:
            if answer == 'y':
                chorale.show(visual_method)
            break

        print("Invalid input!")





if __name__ == "__main__":
    main()


