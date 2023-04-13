
// state variables
let editMode = false;

// crud object for helping with managing and navigating editions
let editionBrowser = {
  postEditions: {},   // dict for storing editions, structured
                      // editionID: {
                      //   title: str
                      //   content: str
                      //}
  nextInsertionID: 1,
  currentEditionID: 0,

  getCurrentEditionID() {
    return this.currentEditionID;
  },
  getCurrentEdition() { 
    return this.postEditions[this.currentEditionID]; 
  },
  getEditionIDs() {
    return Object.keys(this.postEditions).sort(function(a, b){return a-b});
  },
  getAllEditions() {
    return Object.values(this.postEditions);
  },
  getKeyIndexOfCurrentlySelectedEdition() {
    // returns index in terms of keys which have been sorted least to greatest 
    //  based on where the currentEditionID Edtion would sit
    let sortedKeys = Object.keys(this.postEditions).sort(function(a, b){return a-b});
    return sortedKeys.indexOf(this.currentEditionID.toString());
  },
  changeToNextEdition() {
    let sortedKeys = Object.keys(this.postEditions).sort(function(a, b){return a-b});
    indexOfNextKey = (sortedKeys.indexOf(this.currentEditionID) + 1) % sortedKeys.length;
    this.currentEditionID = sortedKeys[indexOfNextKey];
  },
  changeToPrevEdition() {
    let sortedKeys = Object.keys(this.postEditions).sort(function(a, b){return a-b});
    indexOfPrevKey = (sortedKeys.indexOf(this.currentEditionID) - 1 + sortedKeys.length) % sortedKeys.length;
    this.currentEditionID = sortedKeys[indexOfPrevKey];
  },
  addEdition(edition) {
    this.postEditions[this.nextInsertionID] = edition;
    this.currentEditionID = this.nextInsertionID;
    this.nextInsertionID += 1;
  },
  removeCurrentEdition() {
    delete this.postEditions[currentEditionID];
    currentEditionID -= 1;
  }
}

// components
let prompt = document.getElementById("prompt");
let editBtn = document.getElementById("edit-button");
let enterBtn = document.getElementById("enter-button");
let postBtn = document.getElementById("post-button");
let modalPostBtn = document.getElementById("modal-post-button");
let carouselPrevBtn = document.getElementById("carousel-prev");
let carouselNextBtn = document.getElementById("carousel-next");

// display updaters
const setFullPostViewData = (titleStr, bodyStr) => {

  // set the title and body
  let modalTitle = document.getElementById("full-post-view-title")
  let modalBodyText = document.getElementById("full-post-view-body-text")
  modalTitle.textContent = titleStr;
  modalBodyText.textContent = bodyStr;
}

const addSpinner = (htmlRootElementId) => {
  // helper to show spinning icon during requests
  let rootElement = document.getElementById(htmlRootElementId)

  let html = `
  <div id="loading-spinner" class="spinner-border" role="status">
    <span class="visually-hidden">Loading...</span>
  </div>`;
  rootElement.innerHTML = html;
}

const fillPostModal = (edition) => {
  let modalTitleElement = document.getElementById("post-modal-title");
  let modalContentElement = document.getElementById("post-model-content");

  modalTitleElement.textContent = edition.title;
  modalContentElement.textContent = edition.content;
}

const updateEditionTitle = (editionBrowser) => {
  let currEdnID = editionBrowser.getCurrentEditionID();
  let titleStr = `Edition #${currEdnID}`
  let includeStr = `@edition${currEdnID}`

  let carouselTitle = document.getElementById("carousel-title");
  carouselTitle.textContent =  titleStr;

  // include edition tag if there is a valid edition availble
  if (currEdnID + 1 > 0) {
    let carouselTitleIncludeTag = document.getElementById("carousel-title-include-tag");
    carouselTitleIncludeTag.textContent = "Include edition with: " + includeStr;
  }
}

const updateTitleToLoadStatus = () => {
  let carouselTitle = document.getElementById("carousel-title");
  let titleStr = "Loading New Edition";
  carouselTitle.textContent =  titleStr;
}

const updateEditionView = (editionBrowser) => {
  let contentSection = document.getElementById("carousel-content-section");
  let currEditions = editionBrowser.getAllEditions();
  let indexOfCurrEdition = editionBrowser.getKeyIndexOfCurrentlySelectedEdition();
  // let currEditionID = editionBrowser.getCurrentEditionID();

  console.log("index of curr edition", indexOfCurrEdition);

  updateEditionTitle(editionBrowser);

  let html = "";
  for (let i = 0; i < currEditions.length; i++) {
    html += `
      <div class="carousel-item ${i === indexOfCurrEdition ? 'active' : ''}">
        <h4 class="card-title">${currEditions[i].title}</h4>
        <p class="card-text">${currEditions[i].content}</p>
      </div>
    `;
  }
  contentSection.innerHTML = html;
}

const switchToEditMode = () => {
  let contentSection = document.getElementById("carousel-content-section");
  const currentEdition = postEditions[currentEditionID]

  let html = "";
  html += `
      <div class="carousel-item active container">
        <div class="row">
          <label for="" 
          <textarea id="" style="width: 100%;">${currentEdition.title}</textarea>
        </div>
        <textarea class="row">${currentEdition.content}</textarea>
      </div>
    `;

  contentSection.innerHTML = html;
}

const switchToViewMode = () => {

}

// event listners
enterBtn.onclick = () => {
  const promptContent = prompt.value;                      // get the current prompt in the text box 
  const currentEditions = editionBrowser.getAllEditions(); // get the currently saved editions

  // clear text box
  prompt.value = "";

  // make api request to flask model wrapper endpoint
  sendPromptToOpenAI(promptContent, currentEditions)
    .then(addSpinner("carousel-content-section"))
    .then(updateTitleToLoadStatus())
    .then(modelResponse => {
      // create a new edition
      const newEdition = {
        title: modelResponse.title,
        content: modelResponse.content
      }

      editionBrowser.addEdition(newEdition);

      // postEditions[currentEditionID] = newEdition;
      // currentEditionID = postEditions.length - 1;

      updateEditionView(editionBrowser);
      updateEditionTitle(editionBrowser);
    });
}

carouselNextBtn.addEventListener("click", () => {
  editionBrowser.changeToNextEdition();

  let carouselTitle = document.getElementById("carousel-title");
  let currEdnID = editionBrowser.getCurrentEditionID();
  carouselTitle.textContent = `Edition #${currEdnID}`;
});
carouselPrevBtn.addEventListener("click", () => {
  editionBrowser.changeToPrevEdition();

  let carouselTitle = document.getElementById("carousel-title");
  let currEdnID = editionBrowser.getCurrentEditionID();
  carouselTitle.textContent = `Edition #${currEdnID}`;
});



editBtn.onclick = () => {
  switchToEditMode();
}
carouselPrevBtn.onclick = () => {editMode = false}
carouselNextBtn.onclick = () => {editMode = false}


postBtn.onclick = () => { 
  // fill with the current edition
  let currEdition = editionBrowser.getCurrentEdition();
  fillPostModal(currEdition);
}

modalPostBtn.onclick = () => {
  // post current edition to selected area 
  // for now just reddit
  let modalSubredditPrompt = document.getElementById("post-address-prompt");
  let modalBody = document.getElementById("post-modal-body");

  let currEdition = editionBrowser.getCurrentEdition();
    postEditionToReddit(modalSubredditPrompt.value, currEdition)
    .then(addSpinner("carousel-content-section"))
    .then(addSpinner("post-modal-body"))
    .then(modalResponse => {
      modalBody.innerHTML = `
        <b>Post Success</b>
      `
    })
    .catch(error => console.log('error:', error));;
}