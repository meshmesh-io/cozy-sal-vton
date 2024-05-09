/* global document, Image */
// eslint-disable-next-line import/no-unresolved
import { app } from '/scripts/app.js';
// eslint-disable-next-line import/no-unresolved
import { api } from '/scripts/api.js';
// eslint-disable-next-line import/no-unresolved
import { $el } from '/scripts/ui.js';

const LANDMARK_IDS = [
  'tip of back',
  'lefmost neck',
  'next to leftmost neck',
  'neck bottom',
  'neck bottom',
  'next to rightmost neck',
  'rightmost neck',
  'left shoulder',
  'left armpit',
  'upper left arm exterior',
  'upper left arm interior',
  'left elbow exterior',
  'left elbow interior',
  'lower left arm exterior',
  'lower left arm interior',
  'end of left sleeve exterior',
  'end of left sleeve interior',
  'right shoulder',
  'right armpit',
  'upper right arm exterior',
  'upper right arm interior',
  'right elbow exterior',
  'right elbow interior',
  'lower right arm exterior',
  'lower right arm interior',
  'end of right sleeve exterior',
  'end of right sleeve interior',
  'left waist',
  'right waist',
  'left garment bottom',
  'middle garment bottom',
  'right garment bottom',
];

class SALVTON {
  constructor(node) {
    this.serializedCtx = {};
    this.node = node;
    this.node.properties = this.node.properties || {};

    const pkw = this.node.widgets?.find((w) => w.name === 'person_landmarks');
    const gkw = this.node.widgets?.find((w) => w.name === 'garment_landmarks');

    let personImage = '';
    let garmentImage = '';
    let personData = [];
    let garmentData = [];
    let selectedLandmark = -1;

    const importFromInputs = () => {
      personData = [];
      pkw.value?.split('\n').forEach((line) => {
        if (line.includes('[') && line.includes(']')) {
          personData.push(
            line
              .split('[')[1]
              .split(']')[0]
              .split(', ')
              .map((v) => Number(v))
          );
        }
      });

      garmentData = [];
      gkw.value?.split('\n').forEach((line) => {
        if (line.includes('[') && line.includes(']')) {
          garmentData.push(
            line
              .split('[')[1]
              .split(']')[0]
              .split(', ')
              .map((v) => Number(v))
          );
        }
      });
    };

    const exportToInputs = () => {
      const personLines = pkw.value.split('\n');
      personData.forEach((landmark, idx) => {
        const comment = personLines[idx + 1].split(' # ')[1] || '';
        personLines[idx + 1] =
          `  [${landmark[0]}, ${landmark[1]}, ${landmark[2]}], # ${comment}`;
      });
      pkw.value = personLines.join('\n');

      const garmentLines = gkw.value.split('\n');
      garmentData.forEach((landmark, idx) => {
        const comment = garmentLines[idx + 1].split(' # ')[1] || '';
        garmentLines[idx + 1] =
          `  [${landmark[0]}, ${landmark[1]}, ${landmark[2]}], # ${comment}`;
      });
      gkw.value = garmentLines.join('\n');

      this.node.onResize?.(this.node.size);
    };

    api.addEventListener('salvton-landmarks-update', (data) => {
      if (pkw && data.detail.person) {
        pkw.value = data.detail.person;
      }
      if (gkw && data.detail.garment) {
        gkw.value = data.detail.garment;
      }
      if (data.detail.person_image) {
        personImage = data.detail.person_image;
      }
      if (data.detail.garment_image) {
        garmentImage = data.detail.garment_image;
      }
      this.node.onResize?.(this.node.size);
      importFromInputs();
    });

    const updateCanvas = () => {
      importFromInputs();

      const drawPoint = (ctx, x, y, i, cat) => {
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle =
          selectedLandmark === i
            ? 'green'
            : cat === 0
              ? 'red'
              : cat === 1
                ? 'orange'
                : 'blue';
        ctx.fill();
        ctx.font = '12px serif';
        ctx.fillText(i, x + 10, y);
      };

      const personCanvas = document
        .getElementById('person-canvas')
        .getContext('2d');
      const personImg = new Image();
      personImg.src = `/view?filename=${personImage}&subfolder=&type=temp`;
      personImg.onload = () => {
        personCanvas.drawImage(personImg, 0, 0);
        personData.forEach((landmark, idx) =>
          drawPoint(personCanvas, landmark[0], landmark[1], idx, landmark[2])
        );
      };

      const garmentCanvas = document
        .getElementById('garment-canvas')
        .getContext('2d');
      const garmentImg = new Image();
      garmentImg.src = `/view?filename=${garmentImage}&subfolder=&type=temp`;
      garmentImg.onload = () => {
        garmentCanvas.drawImage(garmentImg, 0, 0);
        garmentData.forEach((landmark, idx) =>
          drawPoint(garmentCanvas, landmark[0], landmark[1], idx, landmark[2])
        );
      };
    };

    this.node.addWidget(
      'button',
      'Update Landmarks',
      null,
      () => {
        const modal = document.createElement('div');
        modal.id = 'salvton-landmarks-modal';
        modal.style.position = 'fixed';
        modal.style.width = '100%';
        modal.style.height = '100%';
        modal.style.left = 0;
        modal.style.top = 0;
        modal.style.zIndex = 100;
        modal.style.backgroundColor = 'rgba(0,0,0,1)';
        modal.innerHTML = `
          <style>
            .salvton-flex-container {
              display: flex;
              height: 100vh;
            }
            .salvton-flex-column {
              padding: 10px;
            }
            .salvton-active-button {
              background-color: #4CAF50;
            }
          </style>
          <div class="salvton-flex-container">
            <div id="landmark-buttons" class="salvton-flex-column" style="min-width: 260px; padding-top: 30px;">
              <button onclick="document.getElementById('salvton-landmarks-modal').remove()">EXIT</button><br/>
            </div>
            <div class="salvton-flex-column">
              <canvas id="person-canvas" width="768" height="1024" style="width: 100%"></canvas>
            </div>
            <div class="salvton-flex-column">
              <canvas id="garment-canvas" width="768" height="1024" style="width: 100%"></canvas>
            </div>
          </div>
        `;
        document.body.appendChild(modal);

        importFromInputs();

        const personCanvas = document.getElementById('person-canvas');
        const garmentCanvas = document.getElementById('garment-canvas');
        personCanvas.addEventListener('mousedown', (e) => {
          if (selectedLandmark !== -1) {
            const scale = personCanvas.width / personCanvas.offsetWidth;
            const rect = personCanvas.getBoundingClientRect();
            const x = parseInt((e.clientX - rect.left) * scale);
            const y = parseInt((e.clientY - rect.top) * scale);
            personData[selectedLandmark][0] = x;
            personData[selectedLandmark][1] = y;
            exportToInputs();
            updateCanvas();
          }
        });
        garmentCanvas.addEventListener('mousedown', (e) => {
          if (selectedLandmark !== -1) {
            const scale = garmentCanvas.width / garmentCanvas.offsetWidth;
            const rect = garmentCanvas.getBoundingClientRect();
            const x = parseInt((e.clientX - rect.left) * scale);
            const y = parseInt((e.clientY - rect.top) * scale);
            garmentData[selectedLandmark][0] = x;
            garmentData[selectedLandmark][1] = y;
            exportToInputs();
            updateCanvas();
          }
        });

        const landmarkButtons = document.getElementById('landmark-buttons');
        LANDMARK_IDS.forEach((id, idx) => {
          $el('button', {
            className: 'salvton-landmark-button',
            textContent: `${idx}: ${id}`,
            parent: landmarkButtons,
            onclick: () => {
              const buttons = document.getElementsByClassName(
                'salvton-landmark-button'
              );
              if (selectedLandmark !== -1) {
                buttons[selectedLandmark].classList.remove(
                  'salvton-active-button'
                );
              }
              if (selectedLandmark === idx) {
                selectedLandmark = -1;
              } else {
                buttons[idx].classList.add('salvton-active-button');
                selectedLandmark = idx;
              }

              updateCanvas();
            },
          });
          $el(
            'select',
            {
              parent: landmarkButtons,
              oninput: (e) => {
                const buttons = document.getElementsByClassName(
                  'salvton-landmark-button'
                );
                if (selectedLandmark !== -1) {
                  buttons[selectedLandmark].classList.remove(
                    'salvton-active-button'
                  );
                }
                selectedLandmark = idx;
                buttons[idx].classList.add('salvton-active-button');
                personData[idx][2] = Number(e.target.value);
                exportToInputs();
                updateCanvas();
              },
            },
            [
              $el('option', {
                value: '0',
                textContent: '0',
                selected: personData[idx][2] === 0,
              }),
              $el('option', {
                value: '1',
                textContent: '1',
                selected: personData[idx][2] === 1,
              }),
              $el('option', {
                value: '2',
                textContent: '2',
                selected: personData[idx][2] === 2,
              }),
            ]
          );

          $el(
            'select',
            {
              parent: landmarkButtons,
              oninput: (e) => {
                const buttons = document.getElementsByClassName(
                  'salvton-landmark-button'
                );
                if (selectedLandmark !== -1) {
                  buttons[selectedLandmark].classList.remove(
                    'salvton-active-button'
                  );
                }
                selectedLandmark = idx;
                buttons[idx].classList.add('salvton-active-button');
                garmentData[idx][2] = Number(e.target.value);
                exportToInputs();
                updateCanvas();
              },
            },
            [
              $el('option', {
                value: '0',
                textContent: '0',
                selected: garmentData[idx][2] === 0,
              }),
              $el('option', {
                value: '1',
                textContent: '1',
                selected: garmentData[idx][2] === 1,
              }),
            ]
          );

          $el('br', { parent: landmarkButtons });
        });

        updateCanvas();
      },
      { serialize: false }
    );
  }
}

app.registerExtension({
  name: 'Cozy SAL-VTON',
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === 'Cozy SAL-VTON Try-on') {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) {
          onNodeCreated.apply(this, []);
        }
        this.SALVTON = new SALVTON(this);
      };
    }
  },
});
