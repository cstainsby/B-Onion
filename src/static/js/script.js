
// state variables
let editMode = false;
let currentEditionID = 0;
let postEditions = [];

// components
let prompt = document.getElementById("prompt");
let editBtn = document.getElementById("edit-button");
let enterBtn = document.getElementById("enter-button");
let carouselPrevBtn = document.getElementById("carousel-prev");
let carouselNextBtn = document.getElementById("carousel-next");

// display updaters
const setFullPostViewData = (titleStr, bodyStr) => {
  console.log("Opening Modal");
  console.log("title " + titleStr);
  console.log("body " + bodyStr);

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

const updateEditionTitle = (titleStr) => {
  let carouselTitle = document.getElementById("carousel-title");
  carouselTitle.textContent =  titleStr;
}

const updateEditionView = () => {
    let contentSection = document.getElementById("carousel-content-section");

    updateEditionTitle(`Edition #${currentEditionID + 1}`)

    let html = "";
    for (let i = 0; i < postEditions.length; i++) {
      html += `
        <div class="carousel-item ${i === currentEditionID ? 'active' : ''}">
          <h4 class="card-title">${postEditions[i].title}</h4>
          <p class="card-text">${postEditions[i].content}</p>
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
  // document.addEventListener('DOMContentLoaded', () => {
  //   // on page load
  //   initializeOpenAI();
  // });


  enterBtn.onclick = () => {
    const promptContent = prompt.value;
    const currentEditions = postEditions;

    // clear text box
    prompt.value = "";

    // make api request to flask model wrapper endpoint
    sendPromptToOpenAI(promptContent, currentEditions)
      .then(addSpinner("carousel-content-section"))
      .then(updateEditionTitle("Loading New Edition"))
      .then(modelResponse => {
        console.log("model res " + modelResponse);
        // create a new edition
        const newEdition = {
          title: modelResponse.title,
          content: modelResponse.content
        }
        
        console.log("new edition " + JSON.stringify(newEdition));
        postEditions.push(newEdition);
        currentEditionID = postEditions.length - 1;

        updateEditionView()
      });
  }

  carouselNextBtn.addEventListener("click", () => {
    let carouselTitle = document.getElementById("carousel-title")
    currentEditionID = (currentEditionID + 1) % postEditions.length;
    carouselTitle.textContent = `Edition #${currentEditionID + 1}`
  });
  carouselPrevBtn.addEventListener("click", () => {
    let carouselTitle = document.getElementById("carousel-title")
    currentEditionID = (currentEditionID - 1 + postEditions.length) % postEditions.length;
    carouselTitle.textContent = `Edition #${currentEditionID + 1}`
  });



  editBtn.onclick = () => {
    console.log("editMode  NOW set to:", editMode)

    switchToEditMode();
  }
  carouselPrevBtn.onclick = () => {editMode = false}
  carouselNextBtn.onclick = () => {editMode = false}