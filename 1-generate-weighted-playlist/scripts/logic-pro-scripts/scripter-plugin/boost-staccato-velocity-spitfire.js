// Joshua Bell Full CC32-to-held-keyswitch translator DEBUG
//
// CC32 values:
//   1   = legato slur
//   2   = legato bow
//   11  = tremolo
//   42  = staccato
//   56  = pizzicato fallback/debug
//   60  = pizzicato
//   71  = trill major 2nd
//   127 = trill minor 2nd

var ARTICULATION_CC = 32;

var CC32_LEGATO_SLUR = 1;
var CC32_LEGATO_BOW = 2;
var CC32_TREMOLO = 11;
var CC32_STACCATO = 42;
var CC32_PIZZICATO_ALT = 56;
var CC32_PIZZICATO = 60;
var CC32_TRILL_MAJOR_2ND = 71;
var CC32_TRILL_MINOR_2ND = 127;

// Logic MIDI note numbers
var KS_TREMOLO_C_SHARP_1 = 37;
var KS_TRILL_MINOR_2ND_D1 = 38;
var KS_TRILL_MAJOR_2ND_D1 = 38;
var KS_STACCATO_D_SHARP_1 = 39;
var KS_PIZZICATO_F_SHARP_1 = 42;
var KS_LEGATO_BOW_C2 = 48;
var KS_LEGATO_SLUR_C_SHARP_2 = 49;

var KEYSWITCH_VELOCITY = 100;

var currentArticulation = CC32_LEGATO_SLUR;
var heldKeyswitchPitch = null;
var heldKeyswitchChannel = null;

var ARTICULATIONS = {};

ARTICULATIONS[CC32_LEGATO_SLUR] = {
  name: "legato slur",
  pitch: KS_LEGATO_SLUR_C_SHARP_2
};

ARTICULATIONS[CC32_LEGATO_BOW] = {
  name: "legato bow",
  pitch: KS_LEGATO_BOW_C2
};

ARTICULATIONS[CC32_TREMOLO] = {
  name: "tremolo",
  pitch: KS_TREMOLO_C_SHARP_1
};

ARTICULATIONS[CC32_STACCATO] = {
  name: "staccato",
  pitch: KS_STACCATO_D_SHARP_1
};

ARTICULATIONS[CC32_PIZZICATO] = {
  name: "pizzicato",
  pitch: KS_PIZZICATO_F_SHARP_1
};

ARTICULATIONS[CC32_PIZZICATO_ALT] = {
  name: "pizzicato ALT",
  pitch: KS_PIZZICATO_F_SHARP_1
};

ARTICULATIONS[CC32_TRILL_MAJOR_2ND] = {
  name: "trill major 2nd",
  pitch: KS_TRILL_MAJOR_2ND_D1
};

ARTICULATIONS[CC32_TRILL_MINOR_2ND] = {
  name: "trill minor 2nd",
  pitch: KS_TRILL_MINOR_2ND_D1
};

function sendKeyswitchOn(pitch, channel) {
  var noteOn = new NoteOn();
  noteOn.pitch = pitch;
  noteOn.velocity = KEYSWITCH_VELOCITY;
  noteOn.channel = channel;
  noteOn.send();

  heldKeyswitchPitch = pitch;
  heldKeyswitchChannel = channel;

  Trace("KEYSWITCH ON pitch=" + pitch + " channel=" + channel);
}

function releaseHeldKeyswitch() {
  if (heldKeyswitchPitch === null) {
    return;
  }

  var noteOff = new NoteOff();
  noteOff.pitch = heldKeyswitchPitch;
  noteOff.velocity = 0;
  noteOff.channel = heldKeyswitchChannel;
  noteOff.send();

  Trace("KEYSWITCH OFF pitch=" + heldKeyswitchPitch + " channel=" + heldKeyswitchChannel);

  heldKeyswitchPitch = null;
  heldKeyswitchChannel = null;
}

function holdKeyswitchForArticulation(articulationValue, channel) {
  var mapping = ARTICULATIONS[articulationValue];

  if (!mapping) {
    Trace("NO MAPPING for CC32 value=" + articulationValue + " releasing held keyswitch");
    releaseHeldKeyswitch();
    return;
  }

  Trace(
    "MAPPING CC32 value=" + articulationValue +
    " → " + mapping.name +
    " pitch=" + mapping.pitch +
    " channel=" + channel
  );

  if (heldKeyswitchPitch === mapping.pitch && heldKeyswitchChannel === channel) {
    Trace("KEYSWITCH already held, not resending");
    return;
  }

  releaseHeldKeyswitch();
  sendKeyswitchOn(mapping.pitch, channel);
}

function HandleMIDI(event) {
  if (event instanceof ControlChange) {
    Trace("CC received number=" + event.number + " value=" + event.value + " channel=" + event.channel);

    if (event.number === ARTICULATION_CC) {
      currentArticulation = event.value;
      holdKeyswitchForArticulation(currentArticulation, event.channel);
      return;
    }

    event.send();
    return;
  }

  if (event instanceof NoteOn && event.velocity > 0) {
    Trace(
      "NOTE ON pitch=" + event.pitch +
      " velocity=" + event.velocity +
      " channel=" + event.channel +
      " currentCC32=" + currentArticulation
    );

    holdKeyswitchForArticulation(currentArticulation, event.channel);
    event.send();
    return;
  }

  event.send();
}

function Reset() {
  Trace("RESET");
  releaseHeldKeyswitch();
  currentArticulation = CC32_LEGATO_SLUR;
}