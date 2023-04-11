
// requests 
const initializeOpenAI = async () => {
  const url = '/openai/init';
  const data = { };
  const options = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  };
  
  try {
    const res = await fetch(url, options);
    const result = await res.json();
    return result.openai_response;
  } catch (error) {
    console.error(error);
  }
  return "";
}

const sendPromptToOpenAI = async (promptText, editions) => {
  const url = '/openai/prompt';
  const data = { 
    prompt_contents: promptText,
    editions: editions
  };
  const options = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  };
  
  try {
    const res = await fetch(url, options);
    const result = await res.json();
    // console.log("result", result["choices"][0].text);
    return result;
  } catch (error) {
    console.error(error);
  }
  return "";
}
