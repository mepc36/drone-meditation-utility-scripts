// CSSS CC32/UACC-style to CSSS CC58 translator
//
// Put this Logic Scripter before Kontakt/CSSS.
//
// Incoming standardized CC32 values:
//   1  = arco / legato
//   11 = tremolo
//   42 = staccato
//   56 = pizzicato
//   70 = trill minor 2nd
//   71 = trill major 2nd
//
// Output CSSS CC58 values:
//   6  = sustain / advanced legato
//   21 = staccato
//   31 = pizzicato
//   46 = trills
//   56 = tremolo
//
// Note:
// CSSS uses one Trills articulation. Minor vs major 2nd is determined
// by the notes you write: half-step = minor 2nd, whole-step = major 2nd.

var INPUT_ARTICULATION_CC = 32;
var OUTPUT_CSSS_CC = 58;

// Incoming standardized CC32 values
var CC32_ARCO = 1;
var CC32_TREMOLO = 11;
var CC32_STACCATO = 42;
var CC32_PIZZICATO = 56;
var CC32_TRILL_MINOR_2ND = 70;
var CC32_TRILL_MAJOR_2ND = 71;

// CSSS CC58 values
var CSSS_ADVANCED_LEGATO = 6;
var CSSS_STACCATO = 21;
var CSSS_PIZZICATO = 31;
var CSSS_TRILLS = 46;
var CSSS_TREMOLO = 56;

var MAP = {};

MAP[CC32_ARCO] = {
  name: "arco / advanced legato",
  csssValue: CSSS_ADVANCED_LEGATO
};

MAP[CC32_TREMOLO] = {
  name: "tremolo",
  csssValue: CSSS_TREMOLO
};

MAP[CC32_STACCATO] = {
  name: "staccato",
  csssValue: CSSS_STACCATO
};

MAP[CC32_PIZZICATO] = {
  name: "pizzicato",
  csssValue: CSSS_PIZZICATO
};

MAP[CC32_TRILL_MINOR_2ND] = {
  name: "trill minor 2nd",
  csssValue: CSSS_TRILLS
};

MAP[CC32_TRILL_MAJOR_2ND] = {
  name: "trill major 2nd",
  csssValue: CSSS_TRILLS
};

function sendCSSSCC58(value, channel) {
  var cc = new ControlChange();
  cc.number = OUTPUT_CSSS_CC;
  cc.value = value;
  cc.channel = channel;
  cc.send();

  Trace("SENT CSSS CC58 value=" + value + " channel=" + channel);
}

function HandleMIDI(event) {
  if (event instanceof ControlChange) {
    Trace(
      "CC received number=" + event.number +
      " value=" + event.value +
      " channel=" + event.channel
    );

    if (event.number === INPUT_ARTICULATION_CC) {
      var mapping = MAP[event.value];

      if (!mapping) {
        Trace("NO CSSS MAPPING for CC32 value=" + event.value);
        return;
      }

      Trace(
        "MAPPING CC32 value=" + event.value +
        " → " + mapping.name +
        " → CSSS CC58 value=" + mapping.csssValue
      );

      sendCSSSCC58(mapping.csssValue, event.channel);
      return;
    }

    // Let CC1, CC11, vibrato, expression, etc. pass through.
    event.send();
    return;
  }

  // Important: do NOT filter low notes.
  // CSSS cello/viola need their full playable ranges.
  event.send();
}

function Reset() {
  Trace("RESET");
}