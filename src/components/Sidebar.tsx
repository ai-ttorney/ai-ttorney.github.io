import { Thread } from '../types/chatTypes';
import { isToday, isYesterday, isThisWeek } from 'date-fns';

interface SidebarProps {
  threads: Thread[];
  activeThread: string | null;
  handleCreateThread: () => void;
  handleDeleteThread: (threadId: string) => void;
  handleLoadThread: (threadId: string) => void;
  handleDownloadChat: (threadId: string) => void;
  t: any;
}


function groupThreadsByDate(threads: Thread[]) {
  const groups: { [key: string]: Thread[] } = {
    Today: [],
    Yesterday: [],
    'Last Week': [],
    Older: [],
  };

  threads.forEach(thread => {
    const date = new Date(thread.last_updated);
    if (isToday(date)) {
      groups.Today.push(thread);
    } else if (isYesterday(date)) {
      groups.Yesterday.push(thread);
    } else if (isThisWeek(date, { weekStartsOn: 1 })) {
      groups['Last Week'].push(thread);
    } else {
      groups.Older.push(thread);
    }
  });

  return groups;
}

function Sidebar({
  threads,
  activeThread,
  handleCreateThread,
  handleDeleteThread,
  handleLoadThread,
  handleDownloadChat,
  t,
}: SidebarProps) {
  const groupedThreads = groupThreadsByDate(threads);

  return (
    <div className="sidebar-wrapper">
      <div className="sidebar-container">
        <button className="sidebar-toggle">☰</button>
  
        <div className="sidebar">
          <button onClick={handleCreateThread} className="new-thread-btn">
            {t.newChat}
          </button>
  
          <div className="thread-list">
            {Object.entries(groupedThreads).map(([group, groupThreads]) =>
              groupThreads.length > 0 && (
                <div key={group}>
                  <div className="thread-group-header">{group}</div>
                  {groupThreads.map((thread) => (
                    <div
                      key={thread.thread_id}
                      className={`thread-item ${thread.thread_id === activeThread ? 'active' : ''}`}
                    >
                      <div
                        className="thread-content"
                        onClick={() => handleLoadThread(thread.thread_id)}
                      >
                        <div className="thread-title">{thread.title}</div>
                        <div className="thread-preview">
                          {new Date(thread.last_updated).toLocaleDateString()}
                        </div>
                      </div>
                      <button
                        className="download-thread-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDownloadChat(thread.thread_id);
                        }}
                        title="Download"
                      >
                        ⭳
                      </button>
                      <button
                        className="delete-thread-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (window.confirm(t.deleteConfirm)) {
                            handleDeleteThread(thread.thread_id);
                          }
                        }}
                        title="Delete"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )
            )}
          </div>
        </div>
      </div>
    </div>
  );
  
}

export default Sidebar;
